#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator/estimator.h"
#include "utility/estimator_parameters.h"
#include "utility/visualization.h"

Estimator estimator;

std::condition_variable con;
queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::PointCloudConstPtr> relo_buf;

std::mutex m_buf;
std::mutex m_state;
std::mutex i_buf;
std::mutex m_estimator;

/**
 * @brief IMU 基本参数 [P,Q,V,Ba,Bg,a,g]
 * P：位置（Position），表示物体在三维空间中的位置坐标，通常以米（m）为单位。
 * Q：四元数（Quaternion），表示物体在三维空间中的姿态（旋转方向和角度），可以用于将本体坐标系（Body Frame）与惯性坐标系（Inertial Frame）之间的转换。
 * R：旋转矩阵（Rotation Matrix），表示物体在三维空间中的旋转姿态，可以用于将本体坐标系（Body Frame）与惯性坐标系（Inertial Frame）之间的转换。
 * V：速度（Velocity），表示物体在三维空间中的速度矢量，通常以米/秒（m/s）为单位。
 * Ba：加速度计偏置（Accelerometer Bias），表示加速度计读数在静止状态下的偏差误差。
 * Bg：陀螺仪偏置（Gyroscope Bias），表示陀螺仪读数在静止状态下的偏差误差。
 * a：线性加速度（Linear Acceleration），表示物体在三个轴向上的加速度，通常以米/秒^2（m/s^2）为单位。
 * g：角速度（Angular Velocity），表示物体在三个轴向上的角速度，通常以弧度/秒（rad/s）为单位。
 */
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_P;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;

bool first_feature = true;
bool first_imu     = true;
double latest_time;
double last_imu_t   = 0;
double current_time = -1;
int sum_of_wait     = 0;

// 从IMU测量值imu_msg和上一个P、Q、V递推得到下一个tmp_P、tmp_Q、tmp_V、中值积分
void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    if (first_imu)
    {
        latest_time = t;
        first_imu   = false;
        return;
    }
    double dt   = t - latest_time;
    latest_time = t;

    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
    tmp_Q                  = tmp_Q * Utility::deltaQ(un_gyr * dt);

    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

// 从估计器中得到滑动窗口当前图像帧的imu更新项[P,Q,V,ba,bg,a,g]
// 对imu_buf中剩余的imu_msg进行PQV递推
void update()
{
    TicToc t_predict;
    latest_time = current_time;
    tmp_P       = estimator.Ps[WINDOW_SIZE];
    tmp_Q       = estimator.Rs[WINDOW_SIZE];
    tmp_V       = estimator.Vs[WINDOW_SIZE];
    tmp_Ba      = estimator.Bas[WINDOW_SIZE];
    tmp_Bg      = estimator.Bgs[WINDOW_SIZE];
    acc_0       = estimator.acc_0;
    gyr_0       = estimator.gyr_0;

    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
    {
        predict(tmp_imu_buf.front());
    }
}

/**
 * @brief 获得对齐后的IMU和图像特征数据
 * @details  img:  i -------- j  -  -------- k
 *           imu:  - jjjjjjjj - j/k kkkkkkkk -  
 *           直到把缓存中的图像特征数据或者IMU数据取完，才能够跳出此函数，并返回数据  
 * @return std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> 
 *         vector(IMUs, img_msg)
 */
std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    while (true)
    {
        if (imu_buf.empty() || feature_buf.empty())
        {
            return measurements;
        }
        // 对齐标准：IMU最后一个数据的时间要大于第一个图像特征数据的时间
        // estimator.td 配置中已知的时间差，IMU和图像特征数据的时间戳不一定完全对齐
        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("[estimator] Wait for imu, only should happen at the beginning");
            sum_of_wait++;
            return measurements;
        }
        // 对齐标准：IMU第一个数据的时间要小于第一个图像特征数据的时间
        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("[estimator] Throw img, only should happen at the beginning");
            feature_buf.pop();
            continue;
        }
        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        feature_buf.pop();

        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        // 一个图像数据(img_msg)，对应多个imu数据
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
        {
            // emplace_back相比push_back能更好地避免内存的拷贝与移动
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }
        // 多放了一个imu数据，因此当前图像帧和下一图像帧会共用这个imu_msg
        IMUs.emplace_back(imu_buf.front());
        if (IMUs.empty())
        {
            ROS_WARN("[estimator] No imu between two image");
        }
        // 一同塞入measurements中
        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}

// imu回调函数，将imu_msg保存到imu_buf，IMU状态递推并发布[P,Q,V,header]
void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    // 正常情况是当前时间戳要大于上一帧时间戳
    if (imu_msg->header.stamp.toSec() <= last_imu_t)
    {
        ROS_WARN("[estimator] IMU message in disorder!");
        return;
    }
    last_imu_t = imu_msg->header.stamp.toSec();

    // 这里m_buf相当于一个临界区，保证同时只有imu_callback和feature_callback的其中一个能够访问imu_buf和feature_buf
    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();
    con.notify_one(); // 对应process线程中的unique_lock<mutex> lk(mtx); con.wait(lk);

    last_imu_t = imu_msg->header.stamp.toSec();

    {
        // 这里就是一个互斥锁，std::lock_guard<std::mutex>会自动解锁（析构时解锁）
        std::lock_guard<std::mutex> lg(m_state);
        predict(imu_msg);
        std_msgs::Header header = imu_msg->header;
        header.frame_id         = "world";
        // IMU直接递推得到新的PQV
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
        {
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
        }
    }
}

// feature回调函数，将feature_msg放入feature_buf
void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    if (first_feature)
    {
        //skip the first detected feature, which doesn't contain optical flow speed
        // 跳过第一帧图像特征数据，因为第一帧图像特征数据中没有光流速度
        first_feature = false;
        return;
    }
    // 这里的琐和上面的imu_callback类似
    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();
    con.notify_one(); // 对应process线程中的unique_lock<mutex> lk(mtx); con.wait(lk);
}

// restart回调函数，收到restart时清空feature_buf和imu_buf，估计器重置，时间重置
void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("[estimator] Restart the estimator!");
        m_buf.lock();
        // 清空feature_buf和imu_buf
        while (!feature_buf.empty())
        {
            feature_buf.pop();
        }
        while (!imu_buf.empty())
        {
            imu_buf.pop();
        }
        m_buf.unlock();
        // 重置估计器
        m_estimator.lock();
        estimator.clearState();
        estimator.setParameter();
        m_estimator.unlock();
        // 重置时间
        current_time = -1;
        last_imu_t   = 0;
    }
    return;
}

//relocalization回调函数，将points_msg放入relo_buf
void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg)
{
    ROS_INFO("[estimator] Relocalization callback!");
    m_buf.lock();
    relo_buf.push(points_msg);
    m_buf.unlock();
}

/**
 * @brief VIO的主线程
 * @details 等待并获取measurements：vector(IMUs, img_msg)，计算dt
 *          estimator.processIMU()进行IMU预积分         
 *          estimator.setReloFrame()设置重定位帧
 *          estimator.processImage()处理图像帧：初始化，紧耦合的非线性优化     
 */
void process()
{
    while (true)
    {
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
        std::unique_lock<std::mutex> lk(m_buf);
        // 等待feature_callback和imu_callback中的notify_one
        // 上面两个接收数据完成就会被唤醒，在提取measurements时互斥锁m_buf会锁住，此时无法接收数据
        con.wait(lk, [&] {
            return (measurements = getMeasurements()).size() != 0;
        });
        lk.unlock();

        m_estimator.lock();
        for (auto &measurement : measurements)
        {
            auto img_msg = measurement.second;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            for (auto &imu_msg : measurement.first)
            {
                double t     = imu_msg->header.stamp.toSec();
                double img_t = img_msg->header.stamp.toSec() + estimator.td;
                // 发送IMU数据进行预积分
                if (t <= img_t)
                {
                    if (current_time < 0)
                    {
                        // 处理第一帧图像的情况
                        current_time = t;
                    }
                    double dt = t - current_time; // imu的时间应该小于（早于）当前时间
                    ROS_ASSERT(dt >= 0);          // ROS_ASSERT宏，如果dt<0，就会打印出错误信息并终止程序
                    current_time = t;
                    dx           = imu_msg->linear_acceleration.x;
                    dy           = imu_msg->linear_acceleration.y;
                    dz           = imu_msg->linear_acceleration.z;
                    rx           = imu_msg->angular_velocity.x;
                    ry           = imu_msg->angular_velocity.y;
                    rz           = imu_msg->angular_velocity.z;
                    // 进行预积分
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
#if DEBUG_PRINT
                    printf("IMU dt: %f; linear: (%f,%f,%f) angular: (%f,%f,%f)\n", dt, dx, dy, dz, rx, ry, rz);
#endif
                }
                else
                {
                    double dt_1  = img_t - current_time; // 图像的时间应该小于当前时间
                    double dt_2  = t - img_t;            // 这里必然是大于0的，imu时间大于（晚于）图像时间
                    current_time = img_t;
                    ROS_ASSERT(dt_1 >= 0);
                    ROS_ASSERT(dt_1 + dt_2 >= 0);     // 即t - current_time必须小于零，图像时间 --长-- 当前时间 -短- imu时间
                    double w1 = dt_2 / (dt_1 + dt_2); // 正数
                    double w2 = dt_1 / (dt_1 + dt_2); // 负数
                    dx        = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                    dy        = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                    dz        = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                    rx        = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry        = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz        = w1 * rz + w2 * imu_msg->angular_velocity.z;
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
#if DEBUG_PRINT
                    printf("IMU dt: %f; linear: (%f,%f,%f) angular: (%f,%f,%f)\n", dt, dx, dy, dz, rx, ry, rz);
#endif
                }
            }
            // set relocalization frame
            sensor_msgs::PointCloudConstPtr relo_msg = NULL;
            // 从relo_buf中取出最后一个作为重定位帧
            while (!relo_buf.empty())
            {
                relo_msg = relo_buf.front();
                relo_buf.pop();
            }
            if (relo_msg != NULL)
            {
                vector<Vector3d> match_points;
                double frame_stamp = relo_msg->header.stamp.toSec();
                for (unsigned int i = 0; i < relo_msg->points.size(); i++)
                {
                    Vector3d u_v_id;
                    u_v_id.x() = relo_msg->points[i].x;
                    u_v_id.y() = relo_msg->points[i].y;
                    u_v_id.z() = relo_msg->points[i].z;
                    match_points.push_back(u_v_id);
                }
                Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);
                Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
                Matrix3d relo_r = relo_q.toRotationMatrix();
                int frame_index;
                frame_index = relo_msg->channels[0].values[7];
                estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);
            }

            ROS_DEBUG("[estimator] Processing vision data with timestamp: %f \n", img_msg->header.stamp.toSec());

            TicToc t_process;
            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {
                int v             = img_msg->channels[0].values[i] + 0.5;
                int feature_id    = v / NUM_OF_CAM;
                int camera_id     = v % NUM_OF_CAM;
                double x          = img_msg->points[i].x;
                double y          = img_msg->points[i].y;
                double z          = img_msg->points[i].z;
                double p_u        = img_msg->channels[1].values[i];
                double p_v        = img_msg->channels[2].values[i];
                double velocity_x = img_msg->channels[3].values[i];
                double velocity_y = img_msg->channels[4].values[i];
                ROS_ASSERT(z == 1);
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                image[feature_id].emplace_back(camera_id, xyz_uv_velocity);
            }
            // 处理图像特征
            estimator.processImage(image, img_msg->header);
            double whole_t_process = t_process.toc();
            printStatistics(estimator, whole_t_process);

            std_msgs::Header header = img_msg->header;
            header.frame_id         = "world";
            pubOdometry(estimator, header); // "odometry" 里程计信息PQV
            pubKeyPoses(estimator, header);
            pubCameraPose(estimator, header);
            pubPointCloud(estimator, header);
            pubTF(estimator, header);
            pubKeyframe(estimator);
            if (relo_msg != NULL)
            {
                pubRelocalization(estimator);
            }
            ROS_WARN("[estimator] Process timestamp end(img): %f, at(ros): %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
        }
        m_estimator.unlock();
        m_buf.lock();
        m_state.lock();
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
        {
            update();
        }
        m_state.unlock();
        m_buf.unlock();
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle nh("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(nh);
    estimator.setParameter();
/**
 * @brief EIGEN_DONT_PARALLELIZE
 * Eigen库在默认情况下会自动利用多线程来加速矩阵计算等线性代数操作，这对于大规模数据的处理是非常有用的。
 * 但是，在一些特定的场景下，如单线程环境或某些平台上，多线程操作可能会影响程序的正确性和性能。
 * 
 * Eigen库提供了EIGEN_DONT_PARALLELIZE和EIGEN_HAS_OPENMP两个宏定义，前者用于禁用Eigen库的多线程功能，后者用于启用OpenMP支持。
 * 如果用户需要使用OpenMP支持，可以在编译程序时定义EIGEN_HAS_OPENMP宏，从而开启OpenMP多线程支持。当然
 */
#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("[estimator] EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("[estimator] Waiting for image and imu...");

    registerPub(nh);

    ros::Subscriber sub_imu = nh.subscribe(IMU_TOPIC, 200, imu_callback, ros::TransportHints().tcpNoDelay());
    // 需要feature_tracker节点发布的feature话题
    ros::Subscriber sub_image   = nh.subscribe("/feature_tracker/feature", 100, feature_callback);
    ros::Subscriber sub_restart = nh.subscribe("/feature_tracker/restart", 100, restart_callback);
    // 重定位，需要pose_graph节点发布的match_points话题
    ros::Subscriber sub_relo_points = nh.subscribe("/pose_graph/match_points", 100, relocalization_callback);

    // 创建VIO主线程
    std::thread measurement_process{process};
    ros::spin();

    return 0;
}

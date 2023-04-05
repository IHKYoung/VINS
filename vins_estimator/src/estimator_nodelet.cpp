#include <condition_variable>
#include <cv_bridge/cv_bridge.h>
#include <map>
#include <mutex>
#include <opencv2/core/hal/interface.h>
#include <opencv2/imgproc.hpp>
#include <queue>
#include <ros/ros.h>
#include <set>
#include <string>
#include <thread>

#include "estimator/estimator.h"
#include "feature_tracker/feature_tracker.h"
#include "ros/console_backend.h"
#include "sensor_msgs/image_encodings.h"
#include "utility/parameters.h"
#include "utility/tic_toc.h"
#include "utility/visualization.h"

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

namespace estimator_nodelet_ns
{
class EstimatorNodelet : public nodelet::Nodelet
{
public:
    EstimatorNodelet() = default;

private:
    void onInit() override
    {
        ros::NodeHandle &pn = getPrivateNodeHandle();
        ros::NodeHandle &nh = getMTNodeHandle();
        // Debug, Info, Warn, Error, Fatal
        ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

        readParameters(pn);
        estimator.setParameter();
/**
 * EIGEN_DONT_PARALLELIZE
 * Eigen库在默认情况下会自动利用多线程来加速矩阵计算等线性代数操作，这对于大规模数据的处理是非常有用的。
 * 但是，在一些特定的场景下，如单线程环境或某些平台上，多线程操作可能会影响程序的正确性和性能。
 *
 * Eigen库提供了EIGEN_DONT_PARALLELIZE和EIGEN_HAS_OPENMP两个宏定义，前者用于禁用Eigen库的多线程功能，后者用于启用OpenMP支持。
 * 如果用户需要使用OpenMP支持，可以在编译程序时定义EIGEN_HAS_OPENMP宏，从而开启OpenMP多线程支持。当然
 */
#ifdef EIGEN_DONT_PARALLELIZE
        ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
        ROS_WARN("[estimator_nodelet] Waiting for image(rgbd) and imu...");

        registerPub(nh);

        sub_image = nh.subscribe(IMAGE_TOPIC, 100, &EstimatorNodelet::image_callback, this);
        sub_depth = nh.subscribe(DEPTH_TOPIC, 100, &EstimatorNodelet::depth_callback, this);

        if (USE_IMU)
        {
            sub_imu = nh.subscribe(IMU_TOPIC, 200, &EstimatorNodelet::imu_callback, this, ros::TransportHints().tcpNoDelay());
        }
        // topic from pose_graph, notify if there's relocalization
        // 重定位，需要pose_graph节点发布的match_points话题
        sub_relo_points = nh.subscribe("/pose_graph/match_points", 10, &EstimatorNodelet::relocalization_callback, this);

        dura = std::chrono::milliseconds(2);
        // feature_tracker线程
        trackThread = std::thread(&EstimatorNodelet::process_tracker, this);
        // vio主线程
        processThread = std::thread(&EstimatorNodelet::process, this);
    }

    Estimator estimator;

    // thread relevance
    std::thread trackThread, processThread;
    std::chrono::milliseconds dura;
    std::condition_variable con_tracker;
    std::condition_variable con_estimator;
    std::mutex m_feature;
    std::mutex m_backend;
    std::mutex m_buf;
    std::mutex m_vis;

    // ROS and data buf relevance
    ros::Subscriber sub_imu, sub_relo_points, sub_image, sub_depth;
    queue<sensor_msgs::ImageConstPtr> img_buf;
    queue<sensor_msgs::ImageConstPtr> depth_buf;
    queue<pair<pair<std_msgs::Header, sensor_msgs::ImageConstPtr>, map<int, Eigen::Matrix<double, 7, 1>>>> feature_buf;
    queue<sensor_msgs::PointCloudConstPtr> relo_buf;
    queue<pair<std_msgs::Header, cv::Mat>> vis_img_buf;

    bool init_feature = false;
    bool init_pub     = false;

    // frequency control relevance
    bool first_image_flag   = true;
    double first_image_time = 0;
    double last_image_time  = 0;
    int pub_count           = 1;
    int input_count         = 0;

    // imu relevance
    double last_imu_t = 0;

    void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
    {
        if (imu_msg)
        {
            // 正常情况是当前时间戳要大于上一帧时间戳
            if (imu_msg->header.stamp.toSec() <= last_imu_t)
            {
                ROS_WARN("[estimator_nodelet] IMU message in disorder! --> timestamp: %f", imu_msg->header.stamp.toSec());
                return;
            }

            last_imu_t = imu_msg->header.stamp.toSec();
            Vector3d acc(imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z);
            Vector3d gyr(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);
            estimator.inputIMU(last_imu_t, acc, gyr);
        }
    }

    // 下面的m_buf相当于一个临界区，保证同时只有image_callback、depth_callback和relocalization_callback进行操作
    void image_callback(const sensor_msgs::ImageConstPtr &color_msg)
    {
        m_buf.lock();
        img_buf.emplace(color_msg);
        m_buf.unlock();
        con_tracker.notify_one();
    }

    void depth_callback(const sensor_msgs::ImageConstPtr &depth_msg)
    {
        m_buf.lock();
        depth_buf.emplace(depth_msg);
        m_buf.unlock();
        con_tracker.notify_one();
    }

    void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg)
    {
        m_buf.lock();
        relo_buf.push(points_msg);
        m_buf.unlock();
    }

    void visualizeFeatureFilter(const map<int, Eigen::Matrix<double, 7, 1>> &features, double feature_time)
    {
        cv::Mat vis_img;
        m_vis.lock();
        // 要保证vis_img_buf中的image在feature_time之后
        while (!vis_img_buf.empty())
        {
            if (vis_img_buf.front().first.stamp.toSec() == feature_time)
            {
                vis_img = vis_img_buf.front().second;
                vis_img_buf.pop();
                break;
            }
            else if (vis_img_buf.front().first.stamp.toSec() < feature_time)
            {
                vis_img_buf.pop();
            }
            else
            {
                m_vis.unlock();
                return;
            }
        }
        m_vis.unlock();

        // Show image with tracked points in rviz (by topic pub_match)
        for (auto &feature : features)
        {
            cv::circle(vis_img, cv::Point(feature.second[3], feature.second[4]), 5, cv::Scalar(0, 255, 255), 2);
        }
        pubTrackImg(vis_img);

        // // 基于网格的特征点提取
        // cv::imshow("grids_detector_img", estimator.featureTracker.grids_detector_img);
        // cv::imshow("feature_img", vis_img);
        // cv::waitKey(1);
    }

    // thread: feature tracker
    void process_tracker()
    {
        while (ros::ok())
        {
            {
                sensor_msgs::ImageConstPtr color_msg = nullptr;
                sensor_msgs::ImageConstPtr depth_msg = nullptr;

                // 互斥锁，等待-唤醒机制
                std::unique_lock<std::mutex> locker(m_buf);
                while (img_buf.empty() || depth_buf.empty())
                {
                    con_tracker.wait(locker);
                }

                double time_color = img_buf.front()->header.stamp.toSec();
                double time_depth = depth_buf.front()->header.stamp.toSec();

                if (time_color < time_depth - 0.003)
                {
                    img_buf.pop();
                    ROS_DEBUG("[estimator_nodelet] color earlier than depth: throw color\n");
                }
                else if (time_color > time_depth + 0.003)
                {
                    depth_buf.pop();
                    ROS_DEBUG("[estimator_nodelet] color later than depth: throw depth\n");
                }
                else
                {
                    color_msg = img_buf.front();
                    img_buf.pop();
                    depth_msg = depth_buf.front();
                    depth_buf.pop();
                }
                locker.unlock();

                if (color_msg == nullptr || depth_msg == nullptr)
                {
                    ROS_DEBUG("[estimator_nodelet] time_color = %f, time_depth = %f\n", time_color, time_depth);
                    continue;
                }

                if (first_image_flag)
                {
                    first_image_flag = false;
                    first_image_time = time_color;
                    last_image_time  = time_color;
                    continue;
                }

                // detect unstable camera stream
                // 前后两张图片相差1s 或者 时间戳回退
                if (time_color - last_image_time > 1.0 || time_color < last_image_time)
                {
                    ROS_WARN("[estimator_nodelet] image discontinue! reset the feature tracker!");
                    first_image_flag = true;
                    last_image_time  = 0;
                    pub_count        = 1;

                    ROS_WARN("[estimator_nodelet] restart the estimator!");
                    m_feature.lock();
                    // 重置操作
                    while (!feature_buf.empty())
                    {
                        feature_buf.pop();
                    }
                    m_feature.unlock();
                    m_backend.lock();
                    estimator.clearState();
                    estimator.setParameter();
                    m_backend.unlock();
                    last_imu_t = 0;

                    continue;
                }

                // frequency control
                if (round(1.0 * input_count / (time_color - first_image_time)) > FRONTEND_FREQ)
                {
                    ROS_DEBUG("Skip this frame.%f", 1.0 * input_count / (time_color - first_image_time));
                    continue;
                }
                ++input_count;

                // frequency control
                if (round(1.0 * pub_count / (time_color - first_image_time)) <= FREQ)
                {
                    PUB_THIS_FRAME = true;
                    // reset the frequency control
                    if (abs(1.0 * pub_count / (time_color - first_image_time) - FREQ) < 0.01 * FREQ)
                    {
                        first_image_time = time_color;
                        pub_count        = 0;
                        input_count      = 0;
                    }
                }
                else
                {
                    PUB_THIS_FRAME = false;
                }

                TicToc t_tracking;
                cv_bridge::CvImageConstPtr ptr;
                // 判断图像的编码格式，如果是bgr8，则转换为MONO8
                if (color_msg->encoding == "bgr8")
                {
                    // MONO8可以保存一些图像的元数据，比如图像的时间戳、图像的宽高等
                    sensor_msgs::Image img;
                    img.header = color_msg->header;
                    img.height = color_msg->height;
                    img.width  = color_msg->width;
                    img.is_bigendian = color_msg->is_bigendian; //十六进制数0x12345678在大端序中的存储方式为12 34 56 78（内存地址增大方向）
                    img.step = color_msg->step; // 图像数据的每一行所占用的字节数，即每一行像素数据的内存偏移量
                    img.data = color_msg->data; // 指向图像数据的指针， 可以通过对data + i * step + j的访问来访问第i行第j列的像素数据
                    img.encoding = "bgr8";
                    ptr          = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
                }
                else if (color_msg->encoding == "mono8")
                {
                    // MONO8可以保存一些图像的元数据，比如图像的时间戳、图像的宽高等
                    sensor_msgs::Image img;
                    img.header       = color_msg->header;
                    img.height       = color_msg->height;
                    img.width        = color_msg->width;
                    img.is_bigendian = color_msg->is_bigendian;
                    img.step = color_msg->step; // 图像数据的每一行所占用的字节数，即每一行像素数据的内存偏移量
                    img.data = color_msg->data; // 指向图像数据的指针， 可以通过对data + i * step + j的访问来访问第i行第j列的像素数据
                    img.encoding = "mono8";
                    ptr          = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
                }
                else
                {
                    // 其他图片格式直接转换为MONO8
                    // TODO 可能需要考虑其他图片格式的转换
                    ptr = cv_bridge::toCvCopy(color_msg, sensor_msgs::image_encodings::MONO8);
                }

                if (USE_IMU)
                {
                    Matrix3d &&relative_R = estimator.predictMotion(last_image_time, time_color + estimator.td);
                    estimator.featureTracker.readImage(ptr->image, time_color, relative_R);
                }
                else
                {
                    estimator.featureTracker.readImage(ptr->image, time_color);
                }
                last_image_time = time_color;

                for (unsigned int i = 0;; i++)
                {
                    bool completed = false;
                    completed |= estimator.featureTracker.updateID(i);
                    if (!completed)
                    {
                        break;
                    }
                }
                // 特征点id，矫正后归一化平面的3D点(x,y,z=1)，像素2D点(u,v)，像素的速度(vx,vy)
                if (PUB_THIS_FRAME)
                {
                    pub_count++;
                    std_msgs::Header feature_header = color_msg->header;
                    map<int, Eigen::Matrix<double, 7, 1>> image;

                    auto &un_pts       = estimator.featureTracker.cur_un_pts;
                    auto &cur_pts      = estimator.featureTracker.cur_pts;
                    auto &ids          = estimator.featureTracker.ids;
                    auto &pts_velocity = estimator.featureTracker.pts_velocity;
                    for (unsigned int j = 0; j < ids.size(); j++)
                    {
                        if (estimator.featureTracker.track_cnt[j] > 1)
                        {
                            int p_id = ids[j];
                            geometry_msgs::Point32 p;
                            double x = un_pts[j].x;
                            double y = un_pts[j].y;
                            double z = 1;

                            int v             = p_id * NUM_OF_CAM + 0.5;
                            int feature_id    = v / NUM_OF_CAM;
                            double p_u        = cur_pts[j].x;
                            double p_v        = cur_pts[j].y;
                            double velocity_x = pts_velocity[j].x;
                            double velocity_y = pts_velocity[j].y;

                            ROS_ASSERT(z == 1); // 如果z不等于1，则终止程序
                            Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                            xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                            image[feature_id] = xyz_uv_velocity;
                        }
                    }

                    if (!init_pub)
                    {
                        init_pub = true;
                    }
                    else
                    {
                        if (!init_feature)
                        {
                            // skip the first detected feature, which doesn't contain optical
                            // flow speed
                            init_feature = true;
                            continue;
                        }
                        if (!image.empty())
                        {
                            m_feature.lock();
                            // 就是将color_msg和depth_msg一起放入feature_buf中
                            feature_buf.push(make_pair(make_pair(feature_header, depth_msg), std::move(image)));
                            m_feature.unlock();
                            con_estimator.notify_one();
                        }
                        else
                        {
                            first_image_time = time_color;
                            pub_count        = 0;
                            input_count      = 0;
                            continue;
                        }
                    }

                    // Show image with tracked points in rviz (by topic pub_match)
                    if (SHOW_TRACK)
                    {
                        ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
                        cv::Mat stereo_img(ROW, COL, CV_8UC3);
                        stereo_img      = ptr->image;
                        cv::Mat tmp_img = stereo_img.rowRange(0, ROW);

                        for (unsigned int j = 0; j < estimator.featureTracker.cur_pts.size(); j++)
                        {
                            if (estimator.featureTracker.track_cnt[j] > 1)
                            {
                                double len = std::min(1.0, 1.0 * estimator.featureTracker.track_cnt[j] / WINDOW_SIZE);
                                // 显示追踪状态：红色多表示Good（len的值大），BGR颜色空间
                                cv::circle(tmp_img, estimator.featureTracker.cur_pts[j], 5, cv::Scalar(255 * (1 - len), 0, 255 * len), -1);
                                // 绘制特征点的运动轨迹
                                Vector2d tmp_cur_un_pts(estimator.featureTracker.cur_un_pts[j].x, estimator.featureTracker.cur_un_pts[j].y);
                                Vector2d tmp_pts_velocity(estimator.featureTracker.pts_velocity[j].x,
                                                          estimator.featureTracker.pts_velocity[j].y);
                                Vector3d tmp_prev_un_pts;
                                tmp_prev_un_pts.head(2) = tmp_cur_un_pts - 0.10 * tmp_pts_velocity;
                                tmp_prev_un_pts.z()     = 1;
                                Vector2d tmp_prev_uv;
                                estimator.featureTracker.m_camera->spaceToPlane(tmp_prev_un_pts, tmp_prev_uv);
                                // BGR 蓝色的线条
                                cv::line(tmp_img, estimator.featureTracker.cur_pts[j], cv::Point2f(tmp_prev_uv.x(), tmp_prev_uv.y()),
                                         cv::Scalar(255, 0, 0), 1, 8, 0);
                                // 同时打印特征点的全局ID
                                char name[10];
                                sprintf(name, "%d", estimator.featureTracker.ids[j]);
                                cv::putText(tmp_img, name, estimator.featureTracker.cur_pts[j], cv::FONT_HERSHEY_SIMPLEX, 0.5,
                                            cv::Scalar(0, 0, 0));
                            }
                        }
                        if (USE_IMU)
                        {
                            for (auto &predict_pt : estimator.featureTracker.predict_pts)
                            {
                                cv::circle(tmp_img, predict_pt, 2, cv::Scalar(0, 255, 0), -1);
                            }
                        }

                        m_vis.lock();
                        vis_img_buf.push(make_pair(feature_header, tmp_img));
                        m_vis.unlock();
                    }
                }
                static double whole_process_time = 0;
                static size_t cnt_frame          = 0;
                ++cnt_frame;
                double per_process_time = t_tracking.toc();
                whole_process_time += per_process_time;
                ROS_DEBUG("[estimator_nodelet] average feature tracking costs: %f", whole_process_time / cnt_frame);
                ROS_DEBUG("[estimator_nodelet] feature tracking costs: %f", per_process_time);
            }
            std::this_thread::sleep_for(dura);
        }
    }

    // thread: visual-inertial odometry
    void process()
    {
        while (ros::ok())
        {
            std::unique_lock<std::mutex> locker(m_feature);
            while (feature_buf.empty())
            {
                con_estimator.wait(locker);
            }

            pair<pair<std_msgs::Header, sensor_msgs::ImageConstPtr>, map<int, Eigen::Matrix<double, 7, 1>>> feature_msg(
                std::move(feature_buf.front()));
            feature_buf.pop();
            locker.unlock();

            TicToc t_backend;
            m_backend.lock();
            // set relocalization frame
            sensor_msgs::PointCloudConstPtr relo_msg = nullptr;
            while (!relo_buf.empty())
            {
                relo_msg = relo_buf.front();
                relo_buf.pop();
            }

            if (relo_msg != nullptr)
            {
                vector<Vector3d> match_points;
                double frame_stamp = relo_msg->header.stamp.toSec();
                for (auto point : relo_msg->points)
                {
                    Vector3d u_v_id;
                    u_v_id.x() = point.x;
                    u_v_id.y() = point.y;
                    u_v_id.z() = point.z;
                    match_points.push_back(u_v_id);
                }
                Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);
                Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5],
                                   relo_msg->channels[0].values[6]);
                Matrix3d relo_r = relo_q.toRotationMatrix();
                int frame_index;
                frame_index = relo_msg->channels[0].values[7];
                estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);
            }

            // depth has encoding TYPE_16UC1
            // 深度图都转换成16UC1的格式
            cv::Mat depth_img;
            if (feature_msg.first.second == nullptr)
            {
                depth_img = cv::Mat(ROW, COL, CV_16UC1, cv::Scalar(0));
            }
            else
            {
                if (feature_msg.first.second->encoding == "mono16" || feature_msg.first.second->encoding == "16UC1")
                {
                    // 自动图像编码格转换
                    depth_img = cv_bridge::toCvShare(feature_msg.first.second)->image;
                }
                else if (feature_msg.first.second->encoding == "32FC1")
                {
                    cv::Mat depth_32fc1 = cv_bridge::toCvShare(feature_msg.first.second)->image;
                    // 这里的1000避免32F到16U转换时候的精度丢失
                    depth_32fc1.convertTo(depth_img, CV_16UC1, 1000);
                }
                else
                {
                    // 数字1是一个非零的值，它表示条件始终为真，即断言始终会成功通过
                    ROS_ASSERT_MSG(1, "[estimator_nodelet] Unknown depth encoding!");
                }
            }
            estimator.f_manager.inputDepth(depth_img);

            double feature_time = feature_msg.first.first.stamp.toSec();

            estimator.processImage(feature_msg.second, feature_msg.first.first);

            std_msgs::Header header = feature_msg.first.first;
            header.frame_id         = "map";
            pubOdometry(estimator, header); // "odometry" 里程计信息PQV
            pubTF(estimator, header);       // "extrinsic" 相机到IMU的外参
            pubKeyframe(estimator);         // "keyframe_point"、"keyframe_pose" 关键帧位姿和点云
            if (relo_msg != nullptr)
            {
                pubRelocalization(estimator);
            }

            m_backend.unlock();

            // 轨迹相关
            if (SHOW_TRACK)
            {
                pubKeyPoses(estimator, header);   // "key_poses" 关键点三维坐标
                pubCameraPose(estimator, header); // "camera_pose" 相机位姿
                pubPointCloud(estimator, header); // "history_cloud" 点云信息
                visualizeFeatureFilter(feature_msg.second, feature_time); // 可视化特征处理（使用了网格化处理思想）
            }

            static int cnt_frame             = 0;
            static double whole_process_time = 0;
            double per_process_time          = t_backend.toc();
            cnt_frame++;
            whole_process_time += per_process_time;
            printStatistics(estimator, per_process_time);
            ROS_DEBUG("[estimator_nodelet] average backend costs: %f", whole_process_time / cnt_frame);
            ROS_DEBUG("[estimator_nodelet] backend costs: %f", per_process_time);
            std::this_thread::sleep_for(dura);
        }
    }
};

PLUGINLIB_EXPORT_CLASS(estimator_nodelet_ns::EstimatorNodelet, nodelet::Nodelet)
} // namespace estimator_nodelet_ns
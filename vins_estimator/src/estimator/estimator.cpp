#include "estimator/estimator.h"
#include "utility/visualization.h"
#include <Eigen/src/Core/Matrix.h>
#include <algorithm>
#include <iterator>
#include <string>
#include <utility>

Estimator::Estimator() : f_manager{Rs}
{
    ROS_INFO("[estimator] init begins");
    clearState();
}

void Estimator::setParameter()
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
    }
    f_manager.setRic(ric);
    ProjectionFactor::sqrt_info   = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTdFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td                            = TD;
    g                             = G;

    featureTracker.readIntrinsicParameter(CAM_NAMES);
    // 配置文件中的fisheye 通过鱼眼蒙板来去除边缘噪声
    // 需要注意config下的fisheye_mask.png的尺寸大小，默认是512*512
    if (FISHEYE)
    {
        featureTracker.fisheye_mask = cv::imread(FISHEYE_MASK, 0);
        if (!featureTracker.fisheye_mask.data)
        {
            ROS_INFO("[estimator] load fisheye mask fail");
            ROS_BREAK();
        }
        else
        {
            ROS_INFO("[estimator] load fisheye mask success");
        }
    }
    // 使用网格化思想来处理特征点的提取（使得特征点均匀化）
    // TODO 可以改进（考虑图像的有效部分，比如中间的1/3）
    featureTracker.initGridsDetector();
}

/**
 * @brief 重置状态
 * 清空或初始化滑动窗口中所有的状态量
 */
void Estimator::clearState()
{
    m_imu.lock();
    while (!imu_buf.empty())
    {
        imu_buf.pop();
    }
    m_imu.unlock();

    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
        {
            delete pre_integrations[i];
        }
        pre_integrations[i] = nullptr;

        find_solved[i] = 0;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }
    for (auto &it : all_image_frame)
    {
        if (it.second.pre_integration != nullptr)
        {
            delete it.second.pre_integration;
            it.second.pre_integration = nullptr;
        }
    }
    first_imu         = false;
    sum_of_back       = 0;
    sum_of_front      = 0;
    frame_count       = 0;
    solver_flag       = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();
    td = TD;

    openExEstimation = false;

    delete tmp_pre_integration;

    delete last_marginalization_info;

    tmp_pre_integration       = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur       = false;
    relocalization_info = false;

    drift_correct_r = Matrix3d::Identity();
    drift_correct_t = Vector3d::Zero();

    latest_Q = Eigen::Quaterniond(1, 0, 0, 0);

    init_imu = true;

    prevTime = -1;

    initFirstPoseFlag = false;
}

/**
 * @brief 处理IMU数据
 * IMU预积分，中值积分得到当前PQV作为优化初值
 * @param dt
 * @param linear_acceleration
 * @param angular_velocity
 */
void Estimator::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    if (!first_imu)
    {
        first_imu = true;
        acc_0     = linear_acceleration;
        gyr_0     = angular_velocity;
    }

    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    if (frame_count != 0)
    {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        // if(solver_flag != NON_LINEAR)
        tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);
        // 采用的是中值积分的传播方式
        int j             = frame_count;
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr   = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc   = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

/**
 * @brief 处理图像数据
 *        addFeatureCheckParallax()添加特征点到feature中，计算点跟踪的次数和视差，判断是否是关键帧
 *        判断并进行外参标定
 *        进行视觉惯性联合初始化或基于滑动窗口非线性优化的紧耦合VIO
 * @param image 某帧所有特征点的[camera_id,[x,y,z,u,v,vx,vy]]构成的map，索引为feature_id
 * @param header 某帧图像的头信息
 */
void Estimator::processImage(map<int, Eigen::Matrix<double, 7, 1>> &image, const std_msgs::Header &header)
{
    ROS_DEBUG("[estimator] New image coming ------------------------------------------");
    ROS_DEBUG("[estimator] Adding feature points %lu", image.size());
    // 添加之前检测到的特征点到feature容器中，计算每一个点跟踪的次数，以及它的视差
    // 通过检测两帧之间的视差决定次新帧是否作为关键帧
    if (f_manager.addFeatureCheckParallax(frame_count, image, td))
    {
        marginalization_flag = MARGIN_OLD;
    }
    else
    {
        marginalization_flag = MARGIN_SECOND_NEW;
    }
    ROS_DEBUG("[estimator] this frame is -------------------- %s", marginalization_flag ? "reject" : "accept");
    ROS_DEBUG("[estimator] %s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("[estimator] solving %d", frame_count);
    ROS_DEBUG("[estimator] number of feature: %d", f_manager.getFeatureCount());
    Headers[frame_count] = header.stamp.toSec();

    if (USE_IMU)
    {
        double curTime = header.stamp.toSec() + td;

        while (!IMUAvailable(curTime))
        {
            printf("waiting for IMU ... \r");
            std::chrono::milliseconds dura(2);
            std::this_thread::sleep_for(dura);
        }

        std::vector<pair<double, pair<Eigen::Vector3d, Eigen::Vector3d>>> imu_vector;
        getIMUInterval(prevTime, curTime, imu_vector);
        if (!initFirstPoseFlag)
        {
            initFirstIMUPose(imu_vector);
        }
        for (size_t i = 0; i < imu_vector.size(); i++)
        {
            double dt;
            if (i == 0)
            {
                dt = imu_vector[i].first - prevTime;
            }
            else if (i == imu_vector.size() - 1)
            {
                dt = curTime - imu_vector[i - 1].first;
            }
            else
            {
                dt = imu_vector[i].first - imu_vector[i - 1].first;
            }
            processIMU(dt, imu_vector[i].second.first, imu_vector[i].second.second);
        }
        prevTime = curTime;
    }

    ImageFrame imageframe(image, header.stamp.toSec());
    imageframe.pre_integration = tmp_pre_integration;
    all_image_frame.insert(make_pair(header.stamp.toSec(), imageframe));
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    // 如果没有外参则进行标定
    if (ESTIMATE_EXTRINSIC == 2)
    {
        ROS_INFO("[estimator] calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            // 得到两帧之间归一化特征点
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            // 标定从camera到IMU之间的旋转矩阵
            Matrix3d calib_ric;
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("[estimator] initial extrinsic rotation calib success");
                ROS_WARN_STREAM("[estimator] initial extrinsic rotation: " << endl << calib_ric);
                ric[0]             = calib_ric;
                RIC[0]             = calib_ric;
                ESTIMATE_EXTRINSIC = 1; // 此时有外参了
            }
        }
    }

    TicToc t_optimization;
    if (solver_flag == INITIAL)
    {
        // 使用IMU且不是静止初始化（默认情况）
        if (USE_IMU && !STATIC_INIT)
        {
            // frame_count是滑动窗口中图像帧的数量，一开始初始化为0，滑动窗口总帧数WINDOW_SIZE（默认是10）
            // 确保有足够的frame参与初始化
            if (frame_count == WINDOW_SIZE)
            {
                bool result = false;
                // 有外参且当前帧时间戳大于初始化时间戳0.1秒，就进行初始化操作
                if (ESTIMATE_EXTRINSIC != 2 && (header.stamp.toSec() - initial_timestamp) > 0.1)
                {
                    // 视觉惯性联合初始化（需要初始化）
                    result = initialStructure();
                    // 更新初始化时间戳
                    initial_timestamp = header.stamp.toSec();
                }
                // if init sfm success
                if (result)
                {
                    // 先进行一次滑动窗口非线性优化，得到当前帧与第一帧的位姿
                    solver_flag = NON_LINEAR;
                    solveOdometry();
                    slideWindow();
                    f_manager.removeFailures();
                    ROS_INFO("[estimator] Initialization finish!(no IMU and no Static)");
                    last_R  = Rs[WINDOW_SIZE];
                    last_P  = Ps[WINDOW_SIZE];
                    last_R0 = Rs[0];
                    last_P0 = Ps[0];
                }
                else
                {
                    // 初始化失败则直接滑动窗口
                    slideWindow();
                }
            }
            else
            {
                frame_count++;
            }
        }
        else
        {
            // 需要静止初始化
            f_manager.triangulateWithDepth(Ps, tic, ric);

            if (USE_IMU)
            {
                if (frame_count == WINDOW_SIZE)
                {
                    int i = 0;
                    for (auto &frame_it : all_image_frame)
                    {
                        frame_it.second.R = Rs[i];
                        frame_it.second.T = Ps[i];
                        i++;
                    }
                    if (ESTIMATE_EXTRINSIC != 2)
                    {
                        solveGyroscopeBias(all_image_frame, Bgs);
                        for (int j = 0; j <= WINDOW_SIZE; j++)
                        {
                            pre_integrations[j]->repropagate(Vector3d::Zero(), Bgs[j]);
                        }
                        optimization();
                        updateLatestStates();
                        solver_flag = NON_LINEAR;
                        slideWindow();
                        ROS_INFO("[estimator] Initialization finish!(use IMU and on Static)");
                        last_R  = Rs[WINDOW_SIZE];
                        last_P  = Ps[WINDOW_SIZE];
                        last_R0 = Rs[0];
                        last_P0 = Ps[0];
                    }
                }
            }
            else
            {
                if (frame_count == WINDOW_SIZE)
                {
                    optimization();
                    updateLatestStates();
                    solver_flag = NON_LINEAR;
                    slideWindow();
                    ROS_INFO("[estimator] Initialization finish!(no IMU and on Static)");
                }
            }

            if (frame_count < WINDOW_SIZE)
            {
                frame_count++;
                int prev_frame   = frame_count - 1;
                Ps[frame_count]  = Ps[prev_frame];
                Vs[frame_count]  = Vs[prev_frame];
                Rs[frame_count]  = Rs[prev_frame];
                Bas[frame_count] = Bas[prev_frame];
                Bgs[frame_count] = Bgs[prev_frame];
            }
        }
    }
    else
    {
        TicToc t_solve;
        // 如果不使用IMU，就使用PnP来初始化当前帧的位姿
        if (!USE_IMU)
        {
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
        }

        f_manager.triangulateWithDepth(Ps, tic, ric);
        optimization();
        ROS_DEBUG("[estimator] solver costs: %fms", t_solve.toc());

        set<int> removeIndex;
        movingConsistencyCheck(removeIndex);
        if (SHOW_TRACK)
        {
            for (auto iter = image.begin(), iter_next = image.begin(); iter != image.end(); iter = iter_next)
            {
                ++iter_next;
                auto it = removeIndex.find(iter->first);

                if (it != removeIndex.end())
                {
                    image.erase(iter);
                }
            }
        }
        // 故障检测与恢复,一旦检测到故障，系统将切换回初始化阶段
        // little feature 在最新帧中跟踪的特征数小于某一阈值
        // big IMU acc bias estimation 偏置或外部参数估计有较大的变化
        // big IMU gyr bias estimation
        // big translation 最近两个估计器输出之间的位置或旋转有较大的不连续性
        // big z translation
        // big delta_angle
        if (failureDetection())
        {
            ROS_WARN("[estimator] failure detection!");
            failure_occur = true;
            clearState();
            setParameter();
            ROS_WARN("[estimator] system reboot!");
            return;
        }

        slideWindow();
        f_manager.removeFailures();

        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            key_poses.emplace_back(Ps[i]);
        }

        last_R  = Rs[WINDOW_SIZE];
        last_P  = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
        updateLatestStates();
    }

    static double whole_opt_time = 0;
    static size_t cnt_frame      = 0;
    ++cnt_frame;
    whole_opt_time += t_optimization.toc();
    ROS_DEBUG("[estimator] average optimization  costs: %f", whole_opt_time / cnt_frame);
}

/**
 * @brief 视觉的结构初始化
 *        确保IMU有充分运动激励
 *        relativePose()找到具有足够视差的两帧,由F矩阵恢复R、t作为初始值
 *        sfm.construct() 全局纯视觉SFM 恢复滑动窗口帧的位姿
 *        visualInitialAlign()视觉惯性联合初始化
 * @return true 初始化成功
 */
bool Estimator::initialStructure()
{
    // 通过加速度标准差判断IMU是否有充分运动以初始化
    bool is_imu_excited = false;
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt      = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        aver_g     = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt      = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1)); // 标准差
        // ROS_WARN("[estimator] IMU variation %f!", var);
        if (var < 0.25)
        {
            ROS_INFO("[estimator] IMU excitation not enouth!");
            return false;
        }
        else
        {
            is_imu_excited = true;
        }
    }

    TicToc t_sfm;
    // global sfm
    // 将f_manager中的所有feature保存到存有SFMFeature对象的sfm_f中
    Quaterniond Q[frame_count + 1];
    Vector3d T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f;
    for (auto &it_per_id : f_manager.feature)
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id    = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.emplace_back(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()});
            tmp_feature.observation_depth.emplace_back(imu_j, it_per_frame.depth);
        }
        sfm_f.emplace_back(tmp_feature);
    }
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    // 保证具有足够的视差,由F矩阵恢复Rt
    // 第l帧是从第一帧开始到滑动窗口中第一个满足与当前帧的平均视差足够大的帧，会作为参考帧到下面的全局sfm使用
    // 此处的relative_R，relative_T为当前帧到参考帧（第l帧）的坐标系变换Rt
    if (!relativePose(relative_R, relative_T, l))
    {
        ROS_INFO("[estimator] Not enough features or parallax; Move device around");
        return false;
    }

    // 对窗口中每个图像帧求解sfm问题
    // 得到所有图像帧相对于参考帧的姿态四元数Q、平移向量T和特征点坐标sfm_tracked_points。
    GlobalSFM sfm;
    if (!sfm.construct(frame_count + 1, Q, T, l, relative_R, relative_T, sfm_f, sfm_tracked_points))
    {
        ROS_DEBUG("[estimator] global SFM failed!");
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    // solve pnp for all frame
    // 对于所有的图像帧，包括不在滑动窗口中的，提供初始的RT估计，然后solvePnP进行优化，得到每一帧的姿态
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin();
    for (int i = 0; frame_it != all_image_frame.end(); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if ((frame_it->first) == Headers[i])
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R            = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T            = T[i];
            i++;
            continue;
        }
        if ((frame_it->first) > Headers[i])
        {
            i++;
        }
        // Q和T是图像帧的位姿，而不是求解PNP时所用的坐标系变换矩阵
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = -R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        // 罗德里格斯公式将旋转矩阵转换成旋转向量
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            it             = sfm_tracked_points.find(feature_id);
            if (it != sfm_tracked_points.end())
            {
                Vector3d world_pts = it->second;
                cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                pts_3_vector.push_back(pts_3);
                Vector2d img_pts = id_pts.second.head<2>();
                cv::Point2f pts_2(img_pts(0), img_pts(1));
                pts_2_vector.push_back(pts_2);
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        // 保证特征点数大于5
        if (pts_3_vector.size() < 6)
        {
            ROS_WARN_STREAM("[estimator] pts_3_vector size: " << pts_3_vector.size());
            ROS_DEBUG("[estimator] Not enough points for solve pnp !");
            return false;
        }
        /**
         * @brief bool cv::solvePnP(     求解pnp问题
         *   InputArray  objectPoints,   特征点的3D坐标数组
         *   InputArray  imagePoints,    特征点对应的图像坐标
         *   InputArray  cameraMatrix,   相机内参矩阵
         *   InputArray  distCoeffs,     失真系数的输入向量
         *   OutputArray     rvec,       旋转向量
         *   OutputArray     tvec,       平移向量
         *   bool    useExtrinsicGuess = false,  为真则使用提供的初始估计值
         *   int     flags = SOLVEPNP_ITERATIVE) 采用LM优化
         */
        if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("[estimator] solve pnp fail!");
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp, tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        // 这里也同样需要将坐标变换矩阵转变成图像帧位姿，并转换为IMU坐标系的位姿
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp              = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }

    // Rs Ps ric init
    // 进行视觉惯性联合初始化
    if (visualInitialAlignWithDepth())
    {
        if (!is_imu_excited)
        {
            // 利用加速度平均值估计Bas
            Vector3d sum_a(0, 0, 0);
            for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
            {
                double dt      = frame_it->second.pre_integration->sum_dt;
                Vector3d tmp_a = frame_it->second.pre_integration->delta_v / dt;
                sum_a += tmp_a;
            }
            Vector3d avg_a;
            avg_a = sum_a * 1.0 / ((int)all_image_frame.size() - 1);

            Vector3d tmp_Bas = avg_a - Utility::g2R(avg_a).inverse() * G;
            ROS_WARN_STREAM("[estimator] accelerator bias initial calibration " << tmp_Bas.transpose());
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                Bas[i] = tmp_Bas;
            }
        }
        return true;
    }
    else
    {
        ROS_INFO("[estimator] misalign visual structure with IMU");
        return false;
    }
}

bool Estimator::initialStructureWithDepth()
{
    // check imu observibility
    bool is_imu_excited = false;

    map<double, ImageFrame>::iterator frame_it;
    Vector3d sum_a;
    for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
    {
        double dt      = frame_it->second.pre_integration->sum_dt;
        Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
        sum_a += tmp_g;
    }
    Vector3d aver_g;
    aver_g     = sum_a * 1.0 / ((int)all_image_frame.size() - 1);
    double var = 0;
    for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
    {
        double dt      = frame_it->second.pre_integration->sum_dt;
        Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
        var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
    }
    var = sqrt(var / ((int)all_image_frame.size() - 1)); // 标准差
    // ROS_WARN("[estimator] IMU variation %f!", var);
    if (var < 0.25)
    {
        ROS_INFO("[estimator] IMU excitation not enouth!");
        return false;
    }
    else
    {
        is_imu_excited = true;
    }

    if (visualInitialAlignWithDepth())
    {
        if (!is_imu_excited)
        {
            Vector3d tmp_Bas = aver_g - Utility::g2R(aver_g).inverse() * G;
            ROS_WARN_STREAM("[estimator] accelerator bias initial calibration " << tmp_Bas.transpose());
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                Bas[i] = tmp_Bas;
            }
        }
        return true;
    }
    else
    {
        staticInitialAlignWithDepth();
        return true;
    }

    ROS_INFO("[estimator] misalign visual structure with IMU");
    return false;
}

/**
 * @brief 视觉惯性联合初始化
 *        陀螺仪的偏置校准(加速度偏置没有处理) 计算速度V[0:n] 重力g 尺度s
 *        更新了Bgs后，IMU测量量需要repropagate
 *        得到尺度s和重力g的方向后，需更新所有图像帧在世界坐标系下的Ps、Rs、Vs
 * @return true 成功
 */
bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;
    // solve scale
    // 计算陀螺仪偏置，尺度，重力加速度和速度
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if (!result)
    {
        ROS_ERROR("[estimator] solve g failed!");
        return false;
    }

    // change state
    // 得到所有图像帧的位姿Ps、Rs，并将其置为关键帧
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri                              = all_image_frame[Headers[i]].R;
        Vector3d Pi                              = all_image_frame[Headers[i]].T;
        Ps[i]                                    = Pi;
        Rs[i]                                    = Ri;
        all_image_frame[Headers[i]].is_key_frame = true;
    }
    // 将所有特征点的深度置为-1
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < dep.size(); i++)
    {
        dep[i] = -1;
    }
    f_manager.clearDepth(dep);

    // triangulat on cam pose , no tic
    Vector3d TIC_TMP[NUM_OF_CAM];
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        TIC_TMP[i].setZero();
    }
    ric[0] = RIC[0];
    f_manager.setRic(ric);
    // f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));
    f_manager.triangulateWithDepth(Ps, &(TIC_TMP[0]), &(RIC[0]));

    double s = (x.tail<1>())(0);
    // ROS_DEBUG("[estimator] the scale is %f\n", s);
    // do repropagate here
    // 陀螺仪的偏置bgs改变，重新计算预积分
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    // 将Ps、Vs、depth尺度s缩放
    for (int i = frame_count; i >= 0; i--)
    {
        // Ps转变为第i帧imu坐标系到第0帧imu坐标系的变换
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    }
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if (frame_i->second.is_key_frame)
        {
            kv++;
            // Vs为优化得到的速度
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        // if (it_per_id.used_num < 4)
        // {
        //     continue;
        // }
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
        {
            continue;
        }
        it_per_id.estimated_depth *= s;
    }

    // 通过将重力旋转到z轴上，得到世界坐标系与摄像机坐标系c0之间的旋转矩阵rot_diff
    Matrix3d R0 = Utility::g2R(g);
    double yaw  = Utility::R2ypr(R0 * Rs[0]).x();
    R0          = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g           = R0 * g;
    // Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    // 所有变量从参考坐标系c0旋转到世界坐标系w
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_DEBUG_STREAM("[estimator] g0     " << g.transpose());
    ROS_DEBUG_STREAM("[estimator] my R0  " << Utility::R2ypr(Rs[0]).transpose());

    return true;
}

/**
 * @brief 通过Depth实现静止初始化
 *
 * @return true
 * @return false
 */
bool Estimator::staticInitialAlignWithDepth()
{
    // 利用加速度平均值估计Bgs, Bas, g
    map<double, ImageFrame>::iterator frame_it;
    Vector3d sum_a(0, 0, 0);
    Vector3d sum_w(0, 0, 0);
    for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
    {
        sum_a += frame_it->second.pre_integration->delta_v / frame_it->second.pre_integration->sum_dt;
        Vector3d tmp_w;
        for (auto &gyr_msg : frame_it->second.pre_integration->gyr_buf)
        {
            tmp_w += gyr_msg;
        }
        sum_w += tmp_w / frame_it->second.pre_integration->gyr_buf.size();
    }
    Vector3d avg_a   = sum_a * 1.0 / ((int)all_image_frame.size() - 1);
    Vector3d avg_w   = sum_w * 1.0 / ((int)all_image_frame.size() - 1);
    g                = avg_a.normalized() * G.z();
    Vector3d tmp_Bas = avg_a - g;

    // solveGyroscopeBias(all_image_frame, Bgs);
    ROS_WARN_STREAM("[estimator] gyroscope bias initial calibration " << avg_w.transpose());
    ROS_WARN_STREAM("[estimator] accelerator bias initial calibration " << tmp_Bas.transpose());
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        Bgs[i] = avg_w;
        Bas[i] = tmp_Bas;
    }

    // 陀螺仪的偏置bgs改变，重新计算预积分
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Bas[i], Bgs[i]);
    }

    // 通过将重力旋转到z轴上，得到世界坐标系与摄像机坐标系c0之间的旋转矩阵rot_diff
    Matrix3d R0 = Utility::g2R(g);
    double yaw  = Utility::R2ypr(R0).x();
    R0          = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g           = R0 * G;

    Matrix3d rot_diff = R0;
    // 所有变量从参考坐标系c0旋转到世界坐标系w
    for (int i = 0; i <= frame_count; i++)
    {
        ROS_DEBUG("[estimator] %d farme's Ps is (%f, %f, %f)\n", i, Ps[i].x(), Ps[i].y(), Ps[i].z());
        ROS_DEBUG("[estimator] %d farme's Vs is (%f, %f, %f)\n", i, Vs[i].x(), Vs[i].y(), Vs[i].z());
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_DEBUG_STREAM("[estimator] static g0 " << g.transpose());
    ROS_DEBUG_STREAM("[estimator] my R0     " << Utility::R2ypr(Rs[0]).transpose());

    return true;
}

bool Estimator::visualInitialAlignWithDepth()
{
    TicToc t_g;
    VectorXd x;

    // solve scale
    // 计算陀螺仪偏置，尺度，重力加速度和速度
    solveGyroscopeBias(all_image_frame, Bgs);
    if (!LinearAlignmentWithDepth(all_image_frame, g, x))
    {
        ROS_ERROR("[estimator] solve g failed!");
        return false;
    }

    // do repropagate here
    // 陀螺仪的偏置bgs改变，重新计算预积分
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    // change state
    // 得到所有图像帧的位姿Ps、Rs，并将其置为关键帧
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri                              = all_image_frame[Headers[i]].R;
        Vector3d Pi                              = all_image_frame[Headers[i]].T;
        Ps[i]                                    = Pi;
        Rs[i]                                    = Ri;
        all_image_frame[Headers[i]].is_key_frame = true;
    }

    // do repropagate here
    // 陀螺仪的偏置bgs改变，重新计算预积分
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    // ROS_DEBUG("[estimator] before: (%f, %f, %f)\n", Ps[1].x(), Ps[1].y(), Ps[1].z());
    for (int i = frame_count; i >= 0; i--)
    {
        Ps[i] = Ps[i] - Rs[i] * TIC[0] - (Ps[0] - Rs[0] * TIC[0]);
    }
    // ROS_DEBUG("[estimator] after: (%f, %f, %f)\n", Ps[1].x(), Ps[1].y(), Ps[1].z());

    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if (frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }

    // 通过将重力旋转到z轴上，得到世界坐标系与摄像机坐标系c0之间的旋转矩阵rot_diff
    Matrix3d R0 = Utility::g2R(g);
    double yaw  = Utility::R2ypr(R0 * Rs[0]).x();
    R0          = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g           = R0 * g;
    // Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    // 所有变量从参考坐标系c0旋转到世界坐标系w
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_DEBUG_STREAM("[estimator] g0     " << g.transpose());
    ROS_DEBUG_STREAM("[estimator] my R0  " << Utility::R2ypr(Rs[0]).transpose());

    return true;
}

/**
 * @brief 判断两帧有足够视差30且内点数目大于12则可进行初始化，同时得到R和T
 *        断每帧到窗口最后一帧对应特征点的平均视差是否大于30
 *        solveRelativeRT()通过基础矩阵计算当前帧与第l帧之间的R和T，并判断内点数目是否足够
 * @param relative_R
 * @param relative_T
 * @param l
 * @return true 可以进行初始化
 * @return false 不满足初始化条件
 */
bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    // 寻找第i帧到窗口最后一帧的对应特征点
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        // corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        corres = f_manager.getCorrespondingWithDepth(i, WINDOW_SIZE);
        if (corres.size() > 20)
        {
            // 计算平均视差
            double sum_parallax = 0;
            double average_parallax;
            for (auto &corre : corres)
            {
                Vector2d pts_0(corre.first(0) / corre.first(2), corre.first(1) / corre.first(2));
                Vector2d pts_1(corre.second(0) / corre.second(2), corre.second(1) / corre.second(2));
                // Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                // Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax    = sum_parallax + parallax;
            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            // 判断是否满足初始化条件：视差>30和内点数满足要求
            // 同时返回窗口最后一帧（当前帧）到第l帧（参考帧）的Rt
            if (average_parallax * 460 > 30 && m_estimator.solveRelativeRT_PNP(corres, relative_R, relative_T))
            {
                l = i;
                ROS_DEBUG("[estimator] average_parallax: %f, choose l: %d, and newest frame to triangulate the whole structure",
                          average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

// 三角化求解所有特征点的深度，并进行非线性优化
void Estimator::solveOdometry()
{
    if (frame_count < WINDOW_SIZE)
    {
        return;
    }
    if (solver_flag == NON_LINEAR)
    {
        TicToc t_triangulation;
        f_manager.triangulateWithDepth(Ps, tic, ric);
        // f_manager.triangulate(Ps, tic, ric);
        ROS_DEBUG("[estimator] triangulation costs: %f", t_triangulation.toc());
        optimization();
    }
}

// vector转换成double数组（因为ceres使用数值数组）
// Ps、Rs转变成para_Pose
// Vs、Bas、Bgs转变成para_SpeedBias
void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        if (USE_IMU)
        {
            para_SpeedBias[i][0] = Vs[i].x();
            para_SpeedBias[i][1] = Vs[i].y();
            para_SpeedBias[i][2] = Vs[i].z();

            para_SpeedBias[i][3] = Bas[i].x();
            para_SpeedBias[i][4] = Bas[i].y();
            para_SpeedBias[i][5] = Bas[i].z();

            para_SpeedBias[i][6] = Bgs[i].x();
            para_SpeedBias[i][7] = Bgs[i].y();
            para_SpeedBias[i][8] = Bgs[i].z();
        }
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
    {
        para_Feature[i][0] = dep(i);
    }
    if (ESTIMATE_TD)
    {
        para_Td[0][0] = td;
    }
}

// 数据转换，vector2double的相反过程
// 同时这里为防止优化结果往零空间变化，会根据优化前后第一帧的位姿差进行修正。
void Estimator::double2vector()
{
    // 窗口第一帧之前的位姿
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0     = Utility::R2ypr(last_R0);
        origin_P0     = last_P0;
        failure_occur = false;
    }
    if (USE_IMU)
    {
        // 优化后的位姿
        Vector3d origin_R00 =
            Utility::R2ypr(Quaterniond(para_Pose[0][6], para_Pose[0][3], para_Pose[0][4], para_Pose[0][5]).toRotationMatrix());
        // 求得优化前后的姿态差
        double y_diff = origin_R0.x() - origin_R00.x();
        // TODO
        Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
        if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
        {
            ROS_DEBUG("[estimator] euler singular point!");
            rot_diff =
                Rs[0] * Quaterniond(para_Pose[0][6], para_Pose[0][3], para_Pose[0][4], para_Pose[0][5]).toRotationMatrix().transpose();
        }

        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            Rs[i] =
                rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

            Ps[i] =
                rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0], para_Pose[i][1] - para_Pose[0][1], para_Pose[i][2] - para_Pose[0][2])
                + origin_P0;

            // 与IMU预积分相关的量
            Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0], para_SpeedBias[i][1], para_SpeedBias[i][2]);

            Bas[i] = Vector3d(para_SpeedBias[i][3], para_SpeedBias[i][4], para_SpeedBias[i][5]);

            Bgs[i] = Vector3d(para_SpeedBias[i][6], para_SpeedBias[i][7], para_SpeedBias[i][8]);
        }

        // relative info between two loop frame
        if (relocalization_info)
        {
            Matrix3d relo_r;
            Vector3d relo_t;
            relo_r = rot_diff * Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5]).normalized().toRotationMatrix();
            relo_t = rot_diff * Vector3d(relo_Pose[0] - para_Pose[0][0], relo_Pose[1] - para_Pose[0][1], relo_Pose[2] - para_Pose[0][2])
                     + origin_P0;
            double drift_correct_yaw;
            drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();
            drift_correct_r   = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
            drift_correct_t   = prev_relo_t - drift_correct_r * relo_t;
            relo_relative_t   = relo_r.transpose() * (Ps[relo_frame_local_index] - relo_t);
            relo_relative_q   = relo_r.transpose() * Rs[relo_frame_local_index];
            relo_relative_yaw = Utility::normalizeAngle(Utility::R2ypr(Rs[relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());
#if PRINT_DEBUG
            cout << "[estimator] vins relocalization " << endl;
            cout << "[estimator] vins relative_t: " << relo_relative_t.transpose() << endl;
            cout << "[estimator] vins relative_yaw: " << relo_relative_yaw << endl;
#endif
            relocalization_info = 0;
        }
    }
    else
    {
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            Rs[i] = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

            Ps[i] = Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
        }

        // relative info between two loop frame
        if (relocalization_info)
        {
            Matrix3d relo_r;
            Vector3d relo_t;
            relo_r = Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5]).normalized().toRotationMatrix();
            relo_t = Vector3d(relo_Pose[0], relo_Pose[1], relo_Pose[2]);
            double drift_correct_yaw;
            drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();
            drift_correct_r   = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
            drift_correct_t   = prev_relo_t - drift_correct_r * relo_t;
            relo_relative_t   = relo_r.transpose() * (Ps[relo_frame_local_index] - relo_t);
            relo_relative_q   = relo_r.transpose() * Rs[relo_frame_local_index];
            relo_relative_yaw = Utility::normalizeAngle(Utility::R2ypr(Rs[relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());
#if PRINT_DEBUG
            cout << "[estimator] vins relocalization " << endl;
            cout << "[estimator] vins relative_t: " << relo_relative_t.transpose() << endl;
            cout << "[estimator] vins relative_yaw: " << relo_relative_yaw << endl;
#endif
            relocalization_info = false;
        }
    }
    if (USE_IMU)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            tic[i] = Vector3d(para_Ex_Pose[i][0], para_Ex_Pose[i][1], para_Ex_Pose[i][2]);
            ric[i] =
                Quaterniond(para_Ex_Pose[i][6], para_Ex_Pose[i][3], para_Ex_Pose[i][4], para_Ex_Pose[i][5]).normalized().toRotationMatrix();
        }
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
    {
        dep(i) = para_Feature[i][0];
    }
    f_manager.setDepth(dep);
    if (ESTIMATE_TD && USE_IMU)
    {
        td = para_Td[0][0];
    }
}

/**
 * @brief 检测估计器是否出现故障
 * 满足以下任一条件时，就会跳出判断
 * @return true
 * @return false
 */
bool Estimator::failureDetection()
{
    // 在最新帧中跟踪的特征数小于某一阈值（默认2）
    if (f_manager.last_track_num < 2)
    {
        ROS_WARN("[estimator] little feature: %d", f_manager.last_track_num);
        // 表示该情况下，仍然继续
        // return true;
    }
    // 偏置或外部参数估计有较大的变化
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_WARN("[estimator] big IMU acc bias estimation: %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_WARN("[estimator] big IMU gyr bias estimation: %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    // 最近两个估计器输出之间的位置有较大的不连续性
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        ROS_WARN("[estimator] big translation: %f", (tmp_P - last_P).norm());
        return true;
    }
    // 最近两个估计器输出之间的旋转有较大的不连续性
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        ROS_WARN("[estimator] big z translation: %f", abs(tmp_P.z() - last_P.z()));
        return true;
    }
    Matrix3d tmp_R   = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_WARN("[estimator] big delta_angle: %f", delta_angle);
        // return true;
    }
    return false;
}

/**
 * @brief 基于滑动窗口紧耦合的非线性优化，残差项的构造和求解
 *        添加要优化的变量 (p,v,q,ba,bg) 一共15个自由度，IMU的外参也可以加进来
 *        添加残差，残差项分为4块 先验残差+IMU残差+视觉残差+闭环检测残差
 *        根据倒数第二帧是不是关键帧确定边缘化的结果
 */
void Estimator::optimization()
{
    TicToc t_whole, t_prepare;
    vector2double();

    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    // loss_function = new ceres::HuberLoss(1.0);
    loss_function = new ceres::CauchyLoss(1.0);

    // 优化参数：q、p；v、Ba、Bg
    // 添加ceres参数块
    // 因为ceres用的是double数组，所以在下面用vector2double做类型装换
    // Ps、Rs转变成para_Pose，Vs、Bas、Bgs转变成para_SpeedBias
    for (int i = 0; i < frame_count + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        if (USE_IMU)
        {
            problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS); // v、Ba、Bg参数
        }
    }
    if (!USE_IMU)
    {
        problem.SetParameterBlockConstant(para_Pose[0]);
    }
    // 优化参数：imu与camera外参
    for (auto &i : para_Ex_Pose)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(i, SIZE_POSE, local_parameterization);
        if ((ESTIMATE_EXTRINSIC && frame_count == WINDOW_SIZE && Vs[0].norm() > 0.2) || openExEstimation)
        {
            // ROS_DEBUG("[estimator] estimate extinsic param");
            openExEstimation = true;
        }
        else
        {
            // ROS_DEBUG("[estimator] fix extinsic param");
            problem.SetParameterBlockConstant(i);
        }
    }
    // 优化参数：imu与camera之间的time offset
    if (USE_IMU)
    {
        problem.AddParameterBlock(para_Td[0], 1);
        // 速度过低时，不估计td
        if (!ESTIMATE_TD || Vs[0].norm() < 0.2)
        {
            problem.SetParameterBlockConstant(para_Td[0]);
        }
    }

    // 构建残差
    // 先验残差
    if (last_marginalization_info)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL, last_marginalization_parameter_blocks);
    }

    // 预积分残差
    if (USE_IMU)
    {
        // 预积分残差，总数目为frame_count
        for (int i = 0; i < frame_count; i++)
        {
            int j = i + 1;
            // 两图像帧之间时间过长，不使用中间的预积分
            if (pre_integrations[j]->sum_dt > 10.0)
            {
                continue;
            }
            IMUFactor *imu_factor = new IMUFactor(pre_integrations[j]);
            // 添加残差格式：残差因子，鲁棒核函数，优化变量（i时刻位姿，i时刻速度与偏置，i+1时刻位姿，i+1时刻速度与偏置）
            problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
        }
    }

    // 重投影残差
    // 重投影残差相关，此时使用了Huber损失核函数
    int f_m_cnt       = 0;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        if (it_per_id.is_dynamic)
        {
            continue;
        }
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
        {
            continue;
        }
        // if (it_per_id.used_num < 4)
        // {
        //     continue;
        // }
        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        // 遍历观测到路标点的图像帧
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i == imu_j)
            {
                continue;
            }
            Vector3d pts_j = it_per_frame.point; // 测量值
            if (ESTIMATE_TD)
            {
                ProjectionTdFactor *f_td = new ProjectionTdFactor(
                    pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity, it_per_id.feature_per_frame[0].cur_td,
                    it_per_frame.cur_td, it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0],
                                         para_Feature[feature_index], para_Td[0]);
                // 0 initial; 1 by depth image; 2 by triangulate
                if (it_per_id.estimate_flag == 1 && FIX_DEPTH)
                {
                    problem.SetParameterBlockConstant(para_Feature[feature_index]);
                }
                else if (it_per_id.estimate_flag == 2)
                {
                    problem.SetParameterUpperBound(para_Feature[feature_index], 0, 2 / DEPTH_MAX_DIST);
                }
            }
            else
            {
                ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0],
                                         para_Feature[feature_index]);
                if (it_per_id.estimate_flag == 1 && FIX_DEPTH)
                {
                    problem.SetParameterBlockConstant(para_Feature[feature_index]);
                }
                else if (it_per_id.estimate_flag == 2)
                {
                    // 防止远处的点深度过小
                    problem.SetParameterUpperBound(para_Feature[feature_index], 0, 2 / DEPTH_MAX_DIST);
                }
            }
            f_m_cnt++;
        }
    }
    ROS_DEBUG("[estimator] visual measurement count: %d", f_m_cnt);
    ROS_DEBUG("[estimator] prepare for ceres: %f", t_prepare.toc());

    // 添加闭环检测残差，计算滑动窗口中与每一个闭环关键帧的相对位姿，这个相对位置是为后面的图优化准备
    if (relocalization_info)
    {
        ROS_INFO("[estimator] set relocalization factor! \n");
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(relo_Pose, SIZE_POSE, local_parameterization);
        int retrive_feature_index = 0;
        int feature_index         = -1;
        for (auto &it_per_id : f_manager.feature)
        {
            if (it_per_id.is_dynamic)
            {
                continue;
            }
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            {
                continue;
            }
            // if (it_per_id.used_num < 4)
            // {
            //     continue;
            // }
            ++feature_index;
            int start = it_per_id.start_frame;
            if (start <= relo_frame_local_index)
            {
                while ((int)match_points[retrive_feature_index].z() < it_per_id.feature_id)
                {
                    retrive_feature_index++;
                }
                if ((int)match_points[retrive_feature_index].z() == it_per_id.feature_id)
                {
                    Vector3d pts_j = Vector3d(match_points[retrive_feature_index].x(), match_points[retrive_feature_index].y(), 1.0);
                    Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                    ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                    problem.AddResidualBlock(f, loss_function, para_Pose[start], relo_Pose, para_Ex_Pose[0], para_Feature[feature_index]);
                    retrive_feature_index++;
                }
            }
        }
    }

    /**
     * @brief ceres优化参数
     * options.linear_solver_type = ceres::DENSE_SCHUR;
     * 设置线性求解器类型为“密集Schur分解”，这是一种基于Schur补的方法，通常用于求解稠密的非线性最小二乘问题。
     * options.num_threads = 4;
     * 设置使用的线程数为 4，这可以提高求解器的并行性能，加速求解速度。
     * options.trust_region_strategy_type = ceres::DOGLEG;
     * 设置信赖域策略类型为“短步长”，这是一种基于牛顿法的优化算法，用于在局部范围内寻找函数的极小值。
     * options.max_num_iterations = NUM_ITERATIONS;
     * 设置最大迭代次数为一个预定义的常量 NUM_ITERATIONS。
     * options.use_explicit_schur_complement = true;
     * 设置是否使用显式的Schur补矩阵，这可以加速计算过程，提高求解速度。
     * options.minimizer_progress_to_stdout = false;
     * 设置是否在标准输出中显示求解进度信息，这里设置为 false，表示不显示求解进度信息。
     * options.use_nonmonotonic_steps = true;
     * 设置是否允许非单调的迭代步骤，这可以在求解复杂问题时提高收敛速度。
     *
     */
    ceres::Solver::Options options;
    options.linear_solver_type            = ceres::DENSE_SCHUR;
    options.num_threads                   = 4;
    options.trust_region_strategy_type    = ceres::DOGLEG;
    options.max_num_iterations            = NUM_ITERATIONS;
    options.use_explicit_schur_complement = true;
    options.minimizer_progress_to_stdout  = false;
    options.use_nonmonotonic_steps        = true;
    if (marginalization_flag == MARGIN_OLD)
    {
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    }
    else
    {
        options.max_solver_time_in_seconds = SOLVER_TIME;
    }
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
#if PRINT_DEBUG
    cout << summary.BriefReport() << endl;
#endif
    ROS_DEBUG("[estimator] Iterations : %d", static_cast<int>(summary.iterations.size()));
    ROS_DEBUG("[estimator] solver costs: %f", t_solver.toc());
    // 防止优化结果在零空间变化，通过固定第一帧的位姿
    double2vector();

    if (frame_count < WINDOW_SIZE)
    {
        return;
    }

    TicToc t_whole_marginalization;
    // 边缘化处理
    // 如果次新帧是关键帧，将边缘化最老帧，及其看到的路标点和IMU数据，将其转化为先验：
    if (marginalization_flag == MARGIN_OLD)
    {
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();

        // 先验部分，基于先验残差，边缘化滑窗中第0帧时刻的状态向量
        // 1、将上一次先验残差项传递给marginalization_info
        if (last_marginalization_info)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0]
                    || last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info =
                new ResidualBlockInfo(marginalization_factor, NULL, last_marginalization_parameter_blocks, drop_set);

            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        // 2、将第0帧和第1帧间的IMU因子IMUFactor(pre_integrations[1])，添加到marginalization_info中
        if (USE_IMU)
        {
            // imu
            // 预积分部分，基于第0帧与第1帧之间的预积分残差，边缘化第0帧状态向量
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                IMUFactor *imu_factor                  = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
                    imu_factor, NULL, vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                    vector<int>{0, 1}); //边缘化 para_Pose[0], para_SpeedBias[0]
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        // 图像部分，基于与第0帧相关的图像残差，边缘化第一次观测的图像帧为第0帧的路标点和第0帧
        // 3、将第一次观测为第0帧的所有路标点对应的视觉观测，添加到marginalization_info中
        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)
            {
                if (it_per_id.is_dynamic)
                {
                    continue;
                }
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (it_per_id.used_num < 4)
                {
                    continue;
                }
                if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                {
                    continue;
                }

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                // 仅处理第一次观测的图像帧为第0帧的情形
                if (imu_i != 0)
                {
                    continue;
                }

                Vector3d pts_i = it_per_id.feature_per_frame[0].point;
                // 对观测到路标点的图像帧的遍历
                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if (imu_i == imu_j)
                    {
                        continue;
                    }

                    Vector3d pts_j = it_per_frame.point;
                    if (ESTIMATE_TD)
                    {
                        ProjectionTdFactor *f_td =
                            new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                   it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                   it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());

                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
                            f_td, loss_function,
                            vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                            vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    else
                    {
                        ProjectionFactor *f                    = new ProjectionFactor(pts_i, pts_j);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
                            f, loss_function,
                            vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]},
                            vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                }
            }
        }
        // 4、计算每个残差，对应的Jacobian，并将各参数块拷贝到统一的内存（parameter_block_data）中
        TicToc t_pre_margin;
        marginalization_info->preMarginalize();
        ROS_DEBUG("[estimator] pre marginalization %f ms", t_pre_margin.toc());
        // 5、多线程构造先验项舒尔补AX=b的结构，在X0处线性化计算Jacobian和残差
        TicToc t_margin;
        marginalization_info->marginalize();
        ROS_DEBUG("[estimator] marginalization %f ms", t_margin.toc());

        // 仅仅改变滑窗double部分地址映射，具体值的通过slideWindow和vector2double函数完成；边缘化仅仅改变A和b，不改变状态向量
        // 由于第0帧观测到的路标点全被边缘化，即边缘化后保存的状态向量中没有路标点；因此addr_shift无需添加路标点
        // 6.调整参数块在下一次窗口中对应的位置（往前移一格），注意这里是指针，后面slideWindow中会赋新值，这里只是提前占座
        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++) // 最老图像帧数据丢弃，从i=1开始遍历
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1]; // i数据保存到i-1指向的地址，滑窗向前移动一格
            if (USE_IMU)
            {
                addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
            }
        }
        for (auto &i : para_Ex_Pose)
        {
            addr_shift[reinterpret_cast<long>(i)] = i;
        }
        if (ESTIMATE_TD)
        {
            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
        }
        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        delete last_marginalization_info;
        last_marginalization_info             = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
    }
    // 如果次新帧不是关键帧
    else // 将次新的图像帧数据边缘化
    {
        if (last_marginalization_info
            && std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks),
                          para_Pose[WINDOW_SIZE - 1]))
        {
            // 1.保留次新帧的IMU测量，丢弃该帧的视觉测量，将上一次先验残差项传递给marginalization_info
            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info)
            {
                vector<int> drop_set; //记录需要丢弃的变量在last_marginalization_parameter_blocks中的索引
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info =
                    new ResidualBlockInfo(marginalization_factor, NULL, last_marginalization_parameter_blocks, drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
            // 2、premargin
            TicToc t_pre_margin;
            ROS_DEBUG("[estimator] begin pre marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("[estimator] end pre marginalization, %f ms", t_pre_margin.toc());
            // 3、marginalize
            TicToc t_margin;
            ROS_DEBUG("[estimator] begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("[estimator] end marginalization, %f ms", t_margin.toc());
            // 4.调整参数块在下一次窗口中对应的位置（去掉次新帧）
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1) // WINDOW_SIZE - 1会被边缘化，不保存
                {
                    continue;
                }
                else if (i == WINDOW_SIZE) // WINDOW_SIZE数据保存到WINDOW_SIZE-1指向的地址
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])]      = para_Pose[i - 1];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])]      = para_Pose[i];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (auto &i : para_Ex_Pose)
            {
                addr_shift[reinterpret_cast<long>(i)] = i;
            }
            if (ESTIMATE_TD)
            {
                addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
            }

            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

            delete last_marginalization_info;
            last_marginalization_info             = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
        }
    }
    ROS_DEBUG("[estimator] whole marginalization costs: %f", t_whole_marginalization.toc());
    ROS_DEBUG("[estimator] whole time for ceres: %f", t_whole.toc());
}

/**
 * @brief 实现滑动窗口all_image_frame的函数
 * 如果次新帧是关键帧，则边缘化最老帧，将其看到的特征点和IMU数据转化为先验信息
 * 如果次新帧不是关键帧，则舍弃视觉测量而保留IMU测量值，从而保证IMU预积分的连贯性
 */
void Estimator::slideWindow()
{
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD) // 边缘化最老的图像帧，即次新的图像帧为关键帧
    {
        double t_0 = Headers[0];
        back_R0    = Rs[0];
        back_P0    = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {
            // 1、滑窗中的数据往前移动一帧；运行结果就是WINDOW_SIZE位置的状态为之前0位置对应的状态
            //  0,1,2...WINDOW_SIZE ——> 1,2...WINDOW_SIZE,0
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Headers[i] = Headers[i + 1];
                Ps[i].swap(Ps[i + 1]);
                Rs[i].swap(Rs[i + 1]);
                if (USE_IMU)
                {
                    std::swap(pre_integrations[i], pre_integrations[i + 1]);

                    dt_buf[i].swap(dt_buf[i + 1]);
                    linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                    angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                    Vs[i].swap(Vs[i + 1]);
                    Bas[i].swap(Bas[i + 1]);
                    Bgs[i].swap(Bgs[i + 1]);
                }
                ++find_solved[i + 1];
                find_solved[i] = find_solved[i + 1];
            }
            // 2、处理前，WINDOW_SIZE位置的状态为之前0位置对应的状态；处理后，WINDOW_SIZE位置的状态为之前WINDOW_SIZE位置对应的状态；之前0位置对应的状态被剔除
            //  0,1,2...WINDOW_SIZE ——> 1,2...WINDOW_SIZE,WINDOW_SIZE
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE]      = Ps[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE]      = Rs[WINDOW_SIZE - 1];
            if (USE_IMU)
            {
                Vs[WINDOW_SIZE]  = Vs[WINDOW_SIZE - 1];
                Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
                Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }
            find_solved[WINDOW_SIZE] = 0;

            // 3、对时刻t_0(对应滑窗第0帧)之前的所有数据进行剔除；即all_image_frame中仅保留滑窗中图像帧0与图像帧WINDOW_SIZE之间的数据
            map<double, ImageFrame>::iterator it_0;
            it_0 = all_image_frame.find(t_0);
            delete it_0->second.pre_integration;
            it_0->second.pre_integration = nullptr;
            for (auto it = all_image_frame.begin(); it != it_0; ++it)
            {
                delete it->second.pre_integration;
                it->second.pre_integration = NULL;
            }
            all_image_frame.erase(all_image_frame.begin(), it_0);
            all_image_frame.erase(t_0);
            slideWindowOld();
        }
    }
    else // 边缘化次新的图像帧，主要完成的工作是数据衔接
    {
        // 0,1,2...WINDOW_SIZE-2, WINDOW_SIZE-1,
        // WINDOW_SIZE ——> 0,,1,2...WINDOW_SIZE-2, WINDOW_SIZE, WINDOW_SIZE
        if (frame_count == WINDOW_SIZE)
        {
            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1]      = Ps[frame_count];
            Rs[frame_count - 1]      = Rs[frame_count];

            find_solved[WINDOW_SIZE] = 0;

            if (USE_IMU)
            {
                for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
                {
                    double tmp_dt                    = dt_buf[frame_count][i];
                    Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                    Vector3d tmp_angular_velocity    = angular_velocity_buf[frame_count][i];

                    pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                    dt_buf[frame_count - 1].push_back(tmp_dt);
                    linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                    angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
                }
                Vs[frame_count - 1]  = Vs[frame_count];
                Bas[frame_count - 1] = Bas[frame_count];
                Bgs[frame_count - 1] = Bgs[frame_count];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }
            slideWindowNew();
        }
    }
}

// real marginalization is removed in solve_ceres()
// 滑动窗口边缘化次新帧时处理特征点被观测的帧号
void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}

// real marginalization is removed in solve_ceres()
// 滑动窗口边缘化最老帧时处理特征点被观测的帧号
void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        // back_R0、back_P0为窗口中最老帧的位姿
        // Rs、Ps为滑动窗口后第0帧的位姿，即原来的第1帧
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
    {
        f_manager.removeBack();
    }
}

/**
 * @brief   进行重定位
 * @param[in]   _frame_stamp    重定位帧时间戳
 * @param[in]   _frame_index    重定位帧索引值
 * @param[in]   _match_points   重定位帧的所有匹配点
 * @param[in]   _relo_t     重定位帧平移向量
 * @param[in]   _relo_r     重定位帧旋转矩阵
 * @return      void
 */
void Estimator::setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r)
{
    relo_frame_stamp = _frame_stamp;
    relo_frame_index = _frame_index;
    match_points.clear();
    match_points = _match_points;
    prev_relo_t  = std::move(_relo_t);
    prev_relo_r  = std::move(_relo_r);
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        if (relo_frame_stamp == Headers[i])
        {
            relo_frame_local_index = i;
            relocalization_info    = true;
            for (int j = 0; j < SIZE_POSE; j++)
                relo_Pose[j] = para_Pose[i][j];
        }
    }
}

void Estimator::inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity)
{
    m_imu.lock();
    imu_buf.push(make_pair(t, make_pair(linearAcceleration, angularVelocity)));
    m_imu.unlock();

    if (solver_flag == Estimator::SolverFlag::NON_LINEAR)
    {
        // predict imu (no residual error)
        m_propagate.lock();
        predict(t, linearAcceleration, angularVelocity);
        m_propagate.unlock();
        pubLatestOdometry(latest_P, latest_Q, latest_V, t);
    }
}

void Estimator::updateLatestStates()
{
    m_propagate.lock();
    latest_time = Headers[frame_count] + td;
    latest_P    = Ps[frame_count];
    latest_Q    = Rs[frame_count];
    latest_V    = Vs[frame_count];
    latest_Ba   = Bas[frame_count];
    latest_Bg   = Bgs[frame_count];

    if (USE_IMU)
    {
        m_imu.lock();
        queue<pair<double, pair<Eigen::Vector3d, Eigen::Vector3d>>> tmp_imu_buf = imu_buf;
        for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
            predict(tmp_imu_buf.front().first, imu_buf.front().second.first, imu_buf.front().second.second);
        m_imu.unlock();
    }
    m_propagate.unlock();
}

Matrix3d Estimator::predictMotion(double t0, double t1)
{
    Matrix3d relative_R = Matrix3d::Identity();
    if (imu_buf.empty())
        return relative_R;

    bool first_imu = true;
    double prev_imu_time;
    Eigen::Vector3d prev_gyr;

    m_imu.lock();
    queue<pair<double, pair<Eigen::Vector3d, Eigen::Vector3d>>> imu_predict_buf = imu_buf;
    m_imu.unlock();
    if (t1 <= imu_predict_buf.back().first)
    {
        while (imu_predict_buf.front().first <= t0)
        {
            imu_predict_buf.pop();
        }
        while (imu_predict_buf.front().first <= t1 && !imu_predict_buf.empty())
        {
            double t                         = imu_predict_buf.front().first;
            Eigen::Vector3d angular_velocity = imu_predict_buf.front().second.second;
            imu_predict_buf.pop();

            if (first_imu)
            {
                prev_imu_time = t;
                first_imu     = false;
                prev_gyr      = angular_velocity;
                continue;
            }
            double dt              = t - prev_imu_time;
            prev_imu_time          = t;
            Eigen::Vector3d un_gyr = 0.5 * (prev_gyr + angular_velocity) - latest_Bg;
            prev_gyr               = angular_velocity;

            // transform the mean angular velocity from the IMU frame to the cam0 frames.
            // Compute the relative rotation.
            Vector3d cam0_angle_axisd = RIC.back().transpose() * un_gyr * dt;
            relative_R *= AngleAxisd(cam0_angle_axisd.norm(), cam0_angle_axisd.normalized()).toRotationMatrix().transpose();
        }
    }

    return relative_R;
}

void Estimator::predict(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity)
{
    if (init_imu)
    {
        latest_time = t;
        init_imu    = false;
        return;
    }
    double dt                = t - latest_time;
    latest_time              = t;
    Eigen::Vector3d un_acc_0 = latest_Q * (acc_0 - latest_Ba) - g;
    Eigen::Vector3d un_gyr   = 0.5 * (gyr_0 + angularVelocity) - latest_Bg;
    latest_Q                 = latest_Q * Utility::deltaQ(un_gyr * dt);
    Eigen::Vector3d un_acc_1 = latest_Q * (linearAcceleration - latest_Ba) - g;
    Eigen::Vector3d un_acc   = 0.5 * (un_acc_0 + un_acc_1);
    latest_P                 = latest_P + dt * latest_V + 0.5 * dt * dt * un_acc;
    latest_V                 = latest_V + dt * un_acc;
}

bool Estimator::IMUAvailable(double t)
{
    if (!imu_buf.empty() && t <= imu_buf.back().first)
    {
        return true;
    }
    else
    {
        return false;
    }
}

void Estimator::initFirstIMUPose(std::vector<pair<double, pair<Eigen::Vector3d, Eigen::Vector3d>>> &imu_vector)
{
    printf("[estimator] init first imu pose\n");
    initFirstPoseFlag = true;
    Eigen::Vector3d averAcc(0, 0, 0);
    int n = (int)imu_vector.size();
    for (auto &i : imu_vector)
    {
        averAcc = averAcc + i.second.first;
    }
    averAcc = averAcc / n;
    printf("[estimator] averge acc: (%f, %f, %f)\n", averAcc.x(), averAcc.y(), averAcc.z());
    Matrix3d R0 = Utility::g2R(averAcc);
    double yaw  = Utility::R2ypr(R0).x();
    R0          = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    Rs[0]       = R0;
    cout << "[estimator] init R0:" << endl << Rs[0] << endl;
}
bool Estimator::getIMUInterval(double t0, double t1, std::vector<pair<double, pair<Eigen::Vector3d, Eigen::Vector3d>>> &imu_vector)
{
    m_imu.lock();
    if (imu_buf.empty())
    {
        printf("[estimator] not receive imu!\n");
        m_imu.unlock();
        return false;
    }
    if (t1 <= imu_buf.back().first)
    {
        while (imu_buf.front().first <= t0)
        {
            imu_buf.pop();
        }
        while (imu_buf.front().first < t1)
        {
            imu_vector.emplace_back(std::move(imu_buf.front()));
            imu_buf.pop();
        }
        imu_vector.emplace_back(imu_buf.front());
        m_imu.unlock();
        return true;
    }
    else
    {
        printf("[estimator] wait for imu...\n");
        m_imu.unlock();
        return false;
    }
}

double Estimator::reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici, Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj,
                                    Vector3d &ticj, double depth, Vector3d &uvi, Vector3d &uvj)
{
    Vector3d pts_w    = Ri * (rici * (depth * uvi) + tici) + Pi;
    Vector3d pts_cj   = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
    Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
    double rx         = residual.x();
    double ry         = residual.y();
    return sqrt(rx * rx + ry * ry);
}

double Estimator::reprojectionError3D(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici, Matrix3d &Rj, Vector3d &Pj,
                                      Matrix3d &ricj, Vector3d &ticj, double depth, Vector3d &uvi, Vector3d &uvj)
{
    Vector3d pts_w  = Ri * (rici * (depth * uvi) + tici) + Pi;
    Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
    return (pts_cj - uvj).norm() / depth;
}

void Estimator::movingConsistencyCheck(set<int> &removeIndex)
{
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
        {
            continue;
        }

        double depth = it_per_id.estimated_depth;
        if (depth < 0)
        {
            continue;
        }

        double err   = 0;
        double err3D = 0;
        int errCnt   = 0;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i != imu_j)
            {
                Vector3d pts_j = it_per_frame.point;
                err += reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], Rs[imu_j], Ps[imu_j], ric[0], tic[0], depth, pts_i, pts_j);
                // for bleeding points
                err3D +=
                    reprojectionError3D(Rs[imu_i], Ps[imu_i], ric[0], tic[0], Rs[imu_j], Ps[imu_j], ric[0], tic[0], depth, pts_i, pts_j);
                errCnt++;
            }
        }
        if (errCnt > 0)
        {
            if (FOCAL_LENGTH * err / errCnt > 10 || err3D / errCnt > 2.0)
            {
                removeIndex.insert(it_per_id.feature_id);
                it_per_id.is_dynamic = true;
            }
            else
            {
                it_per_id.is_dynamic = false;
            }
        }
    }
}

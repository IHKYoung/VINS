#include "utility/parameters.h"

#include <thread>

double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;
double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
double F_THRESHOLD;
double TD, TR;
double ROW, COL;
double DEPTH_MIN_DIST;
double DEPTH_MAX_DIST;

int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
int MAX_CNT;
int MAX_CNT_SET;
int MIN_DIST;
int FREQ;
int SHOW_TRACK;
int EQUALIZE;
int FISHEYE;
int STEREO_TRACK;
int IMAGE_SIZE;
int NUM_GRID_ROWS;
int NUM_GRID_COLS;
int FRONTEND_FREQ;
int USE_IMU;
int NUM_THREADS;
int STATIC_INIT;
int FIX_DEPTH;
unsigned short DEPTH_MIN_DIST_MM;
unsigned short DEPTH_MAX_DIST_MM;
bool PUB_THIS_FRAME;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;
Eigen::Vector3d G{0.0, 0.0, 9.8};
Eigen::Matrix3d Ric;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string IMAGE_TOPIC;
std::string DEPTH_TOPIC;
std::string IMU_TOPIC;
std::string FISHEYE_MASK;
std::string CAM_NAMES;

template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("[parameters] Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("[parameters] Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

// OpenCV FileStorage读写操作 [参考](https://blog.csdn.net/qq_16952303/article/details/80259660)
void readParameters(ros::NodeHandle &n)
{
    std::string config_file;
    config_file = readParam<std::string>(n, "config_file");
    cv::FileStorage fsSettings;
    fsSettings.open(config_file, cv::FileStorage::READ);
    if (!fsSettings.isOpened())
    {
        std::cerr << "[parameters] ERROR: Wrong path to settings" << std::endl;
    }
    // 默认设置是0，可以设置为2-max
    NUM_THREADS = fsSettings["num_threads"];
    if (NUM_THREADS <= 1)
    {
        // std::thread::hardware_concurrency()返回当前系统支持的最大线程数
        NUM_THREADS = std::thread::hardware_concurrency();
    }

    fsSettings["image_topic"] >> IMAGE_TOPIC;
    fsSettings["depth_topic"] >> DEPTH_TOPIC;

    MAX_CNT = fsSettings["max_cnt"];

    MAX_CNT_SET = MAX_CNT;

    MIN_DIST = fsSettings["min_dist"];

    FREQ                         = fsSettings["freq"];
    F_THRESHOLD                  = fsSettings["F_threshold"];
    SHOW_TRACK                   = fsSettings["show_track"];
    EQUALIZE                     = fsSettings["equalize"];
    FISHEYE                      = fsSettings["fisheye"];
    std::string VINS_FOLDER_PATH = readParam<std::string>(n, "vins_folder");
    if (FISHEYE == 1)
    {
        FISHEYE_MASK = VINS_FOLDER_PATH + "config/fisheye_mask.jpg";
    }
    CAM_NAMES = config_file;

    DEPTH_MIN_DIST = fsSettings["depth_min_dist"];
    DEPTH_MAX_DIST = fsSettings["depth_max_dist"];

    DEPTH_MIN_DIST_MM = DEPTH_MIN_DIST * 1000;
    DEPTH_MAX_DIST_MM = DEPTH_MAX_DIST * 1000;

    NUM_GRID_ROWS = fsSettings["num_grid_rows"];
    NUM_GRID_COLS = fsSettings["num_grid_cols"];
    ROS_INFO("[parameters] NUM_GRID_ROWS: %d, NUM_GRID_COLS: %d", NUM_GRID_ROWS, NUM_GRID_COLS);

    FRONTEND_FREQ = fsSettings["frontend_freq"];
    ROS_INFO("[parameters] FRONTEND_FREQ: %d", FRONTEND_FREQ);

    STEREO_TRACK   = false;
    PUB_THIS_FRAME = false;

    if (FREQ == 0)
        FREQ = 100;

    SOLVER_TIME    = fsSettings["max_solver_time"];
    NUM_ITERATIONS = fsSettings["max_num_iterations"];
    MIN_PARALLAX   = fsSettings["keyframe_parallax"];
    ROS_INFO("[parameters] MIN_PARALLAX: %f", MIN_PARALLAX);
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

    std::string OUTPUT_PATH;
    fsSettings["output_path"] >> OUTPUT_PATH;
    VINS_RESULT_PATH = OUTPUT_PATH + "/vins_result_no_loop.csv";
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();

    USE_IMU = fsSettings["imu"];
    ROS_INFO("[parameters] USE_IMU: %d\n", USE_IMU);
    if (USE_IMU)
    {
        fsSettings["imu_topic"] >> IMU_TOPIC;
        printf("[parameters] IMU_TOPIC: %s\n", IMU_TOPIC.c_str());
        ACC_N = fsSettings["acc_n"];
        ACC_W = fsSettings["acc_w"];
        GYR_N = fsSettings["gyr_n"];
        GYR_W = fsSettings["gyr_w"];
        G.z() = fsSettings["g_norm"];
    }

    ROW        = fsSettings["image_height"];
    COL        = fsSettings["image_width"];
    IMAGE_SIZE = ROW * COL;
    ROS_INFO("[parameters] ROW: %f COL: %f ", ROW, COL);

    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
    if (ESTIMATE_EXTRINSIC == 2)
    {
        ROS_WARN("[parameters] Have no prior about extrinsic param, calibrate extrinsic param");
        RIC.emplace_back(Eigen::Matrix3d::Identity());
        TIC.emplace_back(Eigen::Vector3d::Zero());
        EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.txt";
    }
    else
    {
        if (ESTIMATE_EXTRINSIC == 1)
        {
            ROS_WARN("[parameters] Optimize extrinsic param around initial guess!");
            EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.txt";
        }
        if (ESTIMATE_EXTRINSIC == 0)
        {
            ROS_WARN("[parameters] No need to calibrate extrinsic param");
        }

        cv::Mat cv_R, cv_T;
        fsSettings["extrinsicRotation"] >> cv_R;
        fsSettings["extrinsicTranslation"] >> cv_T;
        Eigen::Matrix3d eigen_R;
        Eigen::Vector3d eigen_T;
        cv::cv2eigen(cv_R, eigen_R);
        cv::cv2eigen(cv_T, eigen_T);
        Eigen::Quaterniond Q(eigen_R);
        eigen_R = Q.normalized();
        Ric     = eigen_R; // 没有使用
        RIC.push_back(eigen_R);
        TIC.push_back(eigen_T);
        ROS_INFO_STREAM("[parameters] Extrinsic_R : " << std::endl << RIC[0]);
        ROS_INFO_STREAM("[parameters] Extrinsic_T : " << std::endl << TIC[0].transpose());
    }

    INIT_DEPTH         = 5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;

    TD          = fsSettings["td"];
    ESTIMATE_TD = fsSettings["estimate_td"];
    if (ESTIMATE_TD)
    {
        ROS_INFO_STREAM("[parameters] Unsynchronized sensors, online estimate time offset, initial td: " << TD);
    }
    else
    {
        ROS_INFO_STREAM("[parameters] Synchronized sensors, fix time offset: " << TD);
    }
    // 全局快门和滚动快门
    ROLLING_SHUTTER = fsSettings["rolling_shutter"];
    if (ROLLING_SHUTTER)
    {
        TR = fsSettings["rolling_shutter_tr"];
        ROS_INFO_STREAM("[parameters] rolling shutter camera, read out time per line: " << TR);
    }
    else
    {
        TR = 0;
    }
    // 实现静止初始化
    STATIC_INIT = fsSettings["static_init"];
    if (!fsSettings["fix_depth"].empty())
    {
        FIX_DEPTH = fsSettings["fix_depth"];
    }
    else
    {
        FIX_DEPTH = 1;
    }
    fsSettings.release();
}

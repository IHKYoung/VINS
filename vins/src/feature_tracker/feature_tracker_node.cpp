#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>

#include "feature_tracker/feature_tracker.h"

vector<uchar> r_status;
vector<float> r_err;
queue<sensor_msgs::ImageConstPtr> img_buf;

ros::Publisher pub_img, pub_match;
ros::Publisher pub_restart;

// 根据NUM_OF_CAM的值，初始化了一个FeatureTracker类型的相机实例
FeatureTracker trackerData[NUM_OF_CAM];

double first_image_time;
double last_image_time = 0;
int pub_count          = 1;
bool first_image_flag  = true;
bool init_pub          = false;

void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    // 记录第一帧图像的时间戳
    if (first_image_flag)
    {
        first_image_flag = false;
        first_image_time = img_msg->header.stamp.toSec();
        last_image_time  = img_msg->header.stamp.toSec();
        return;
    }
    // 相机数据流不稳定时，重置特征点跟踪器
    if (img_msg->header.stamp.toSec() - last_image_time > 1.0 || img_msg->header.stamp.toSec() < last_image_time)
    {
        ROS_WARN("[feature_tracker] Image discontinue! Reset the feature tracker!");
        first_image_flag = true;
        last_image_time  = 0;
        pub_count        = 1;
        std_msgs::Bool restart_flag;
        restart_flag.data = true;
        pub_restart.publish(restart_flag);
        return;
    }
    // 相机数据流正常时，记录最后一帧图像的时间戳
    last_image_time = img_msg->header.stamp.toSec();
    // 控制发布频率，PUB_THIS_FRAME为true时，发布当前帧图像
    if (round(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time)) <= FREQ)
    {
        PUB_THIS_FRAME = true;
        // 通过计算“已经发布的数量/当前帧图像的时间戳与第一帧图像的时间戳的差值(数量/时间=频率)”来控制发布频率
        if (abs(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = img_msg->header.stamp.toSec();
            pub_count        = 0;
        }
    }
    else
    {
        PUB_THIS_FRAME = false;
    }

    cv_bridge::CvImageConstPtr ptr;
    // 判断图像的编码格式，如果是bgr8，则转换为MONO8，否则直接转换为MONO8
    if (img_msg->encoding == "bgr8")
    {
        // MONO8可以保存一些图像的元数据，比如图像的时间戳、图像的宽高等
        sensor_msgs::Image img;
        img.header       = img_msg->header;
        img.height       = img_msg->height;
        img.width        = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian; // 是否为大端模式，大端序中，高位字节排在低位字节的前面，十六进制数0x12345678在大端序中的存储方式为12 34 56 78（内存地址增大方向）
        img.step         = img_msg->step;         // 图像数据的每一行所占用的字节数，即每一行像素数据的内存偏移量
        img.data         = img_msg->data;         // 指向图像数据的指针， 可以通过对data + i * step + j的访问来访问第i行第j列的像素数据
        img.encoding     = "bgr8";
        ptr              = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
    {
        // 其他图片格式直接转换为MONO8
        // TODO 可能需要考虑其他图片格式的转换
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
    }

    cv::Mat show_img = ptr->image;
    TicToc t_feature_tracker;
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ROS_DEBUG("[feature_tracker] Processing camera %d", i);
        // 这里的STEREO_TRACK是用来特征点匹配可视化
        if (i != 1 || !STEREO_TRACK)
        {
            trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), img_msg->header.stamp.toSec());
        }
        else
        {
            // if image is too dark or light, trun on equalize to find enough features
            if (EQUALIZE)
            {
                // 通过直方图均衡化来增强图像的对比度
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
            }
            else
            {
                trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
            }
        }

        if (SHOW_UNDISTORTION)
        {
            // 显示去畸变矫正后的特征点
            trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
        }
    }

    for (unsigned int i = 0; ; i++)
    {
        bool completed = false;
        for (int j = 0; j < NUM_OF_CAM; j++)
        {
            if (j != 1 || !STEREO_TRACK)
            {
                // 通过或操作来判断是否完成特征点ID的更新
                completed |= trackerData[j].updateID(i);
            }
            if (!completed)
            {
                break;
            }
        }
    }

    // 控制发布频率的判断会决定这里是否发布
    // 1. 将特征点id，矫正后归一化平面的3D点(x,y,z=1)，像素2D点(u,v)，像素的速度(vx,vy)封装成sensor_msgs::PointCloudPtr类型的feature_points实例中发布到pub_img;
    // 2. 将图像封装到cv_bridge::cvtColor类型的ptr实例中发布到pub_match
    if (PUB_THIS_FRAME)
    {
        pub_count++;
        sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud); // 矫正后归一化平面的3D点
        sensor_msgs::ChannelFloat32 id_of_point;                                // 特征点的id
        sensor_msgs::ChannelFloat32 u_of_point;                                 // 特征点的像素坐标(u,v)
        sensor_msgs::ChannelFloat32 v_of_point;
        sensor_msgs::ChannelFloat32 velocity_x_of_point; // 特征点的像素速度(vx,vy)
        sensor_msgs::ChannelFloat32 velocity_y_of_point;

        feature_points->header          = img_msg->header;
        feature_points->header.frame_id = "world";

        vector<set<int>> hash_ids(NUM_OF_CAM);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            auto &un_pts       = trackerData[i].cur_un_pts;
            auto &cur_pts      = trackerData[i].cur_pts;
            auto &ids          = trackerData[i].ids;
            auto &pts_velocity = trackerData[i].pts_velocity;
            for (unsigned int j = 0; j < ids.size(); j++)
            {
                if (trackerData[i].track_cnt[j] > 1)
                {
                    int p_id = ids[j];
                    hash_ids[i].insert(p_id);
                    geometry_msgs::Point32 p;
                    p.x = un_pts[j].x;
                    p.y = un_pts[j].y;
                    p.z = 1;

                    feature_points->points.push_back(p);
                    id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
                    u_of_point.values.push_back(cur_pts[j].x);
                    v_of_point.values.push_back(cur_pts[j].y);
                    velocity_x_of_point.values.push_back(pts_velocity[j].x);
                    velocity_y_of_point.values.push_back(pts_velocity[j].y);
                }
            }
        }
        feature_points->channels.push_back(id_of_point);
        feature_points->channels.push_back(u_of_point);
        feature_points->channels.push_back(v_of_point);
        feature_points->channels.push_back(velocity_x_of_point);
        feature_points->channels.push_back(velocity_y_of_point);
        ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(), ros::Time::now().toSec());
        // skip the first image; since no optical speed on frist image
        if (!init_pub)
        {
            init_pub = true;
        }
        else
        {
            pub_img.publish(feature_points);
        }
        // 用线条显示特征点的匹配情况
        if (SHOW_TRACK)
        {
            // 这里变成bgr8是为了显示点的不同颜色
            ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
            // cv::Mat stereo_img(ROW * NUM_OF_CAM, COL, CV_8UC3);
            cv::Mat stereo_img = ptr->image;

            for (int i = 0; i < NUM_OF_CAM; i++)
            {
                cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
                cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);
                // 显示追踪状态：红色Good
                for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
                {
                    double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
                    cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
                    // 绘制特征点的运动轨迹
                    Vector2d tmp_cur_un_pts(trackerData[i].cur_un_pts[j].x, trackerData[i].cur_un_pts[j].y);
                    Vector2d tmp_pts_velocity(trackerData[i].pts_velocity[j].x, trackerData[i].pts_velocity[j].y);
                    Vector3d tmp_prev_un_pts;
                    tmp_prev_un_pts.head(2) = tmp_cur_un_pts - 0.10 * tmp_pts_velocity;
                    tmp_prev_un_pts.z()     = 1;
                    Vector2d tmp_prev_uv;
                    trackerData[i].m_camera->spaceToPlane(tmp_prev_un_pts, tmp_prev_uv);
                    cv::line(tmp_img, trackerData[i].cur_pts[j], cv::Point2f(tmp_prev_uv.x(), tmp_prev_uv.y()), cv::Scalar(255, 0, 0), 1, 8, 0);

                    char name[10];
                    sprintf(name, "%d", trackerData[i].ids[j]);
                    cv::putText(tmp_img, name, trackerData[i].cur_pts[j], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                }
            }
            cv::imshow("vis", stereo_img);
            cv::waitKey(1);
            pub_match.publish(ptr->toImageMsg());
        }
    }
    ROS_INFO("whole feature tracker processing costs: %f", t_feature_tracker.toc());
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "feature_tracker");
    ros::NodeHandle nh("~");
    // Debug, Info, Warn, Error, Fatal
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(nh);

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);
    }

    // 配置文件中的fisheye 通过鱼眼蒙板来去除边缘噪声
    // 需要注意config下的fisheye_mask.png的尺寸大小，默认是512*512
    if (FISHEYE)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
            if (!trackerData[i].fisheye_mask.data)
            {
                ROS_INFO("[feature_tracker] Load fisheye mask fail");
                ROS_BREAK();
            }
            else
                ROS_INFO("[feature_tracker] Load fisheye mask success");
        }
    }

    // 配置文件中的image_topic
    // 在ROS中，消息队列（这里的100）允许订阅者处理消息的速度比发布者发布消息的速度慢，因此可以在队列中缓存一定数量的消息以避免消息的丢失。
    ros::Subscriber sub_img = nh.subscribe(IMAGE_TOPIC, 100, img_callback);

    // /feature_tracker/feature 会被 /vins_estimator 订阅
    // 发布的实例是feature_points，跟踪的特征点，给后端优化用
    pub_img = nh.advertise<sensor_msgs::PointCloud>("feature", 1000);
    // 发布的实例是ptr，跟踪的特征点图，给RVIZ用和调试用
    pub_match = nh.advertise<sensor_msgs::Image>("feature_img", 1000);
    // /feature_tracker/restart 会被 /vins_estimator 订阅
    // 发布的实例是restart_flag，重新启动标志位
    pub_restart = nh.advertise<std_msgs::Bool>("restart", 1000);

    // 这一块配合img_callback中的SHOW_TRACK使用
    if (SHOW_TRACK)
    {
        cv::namedWindow("vis", cv::WINDOW_NORMAL);
    }

    ros::spin();
    return 0;
}
#pragma once
#include <ros/ros.h>
#include <opencv2/highgui/highgui.hpp>

extern std::string IMAGE_TOPIC;
extern std::string DEPTH_TOPIC;
extern std::string IMU_TOPIC;
extern std::string FISHEYE_MASK;
extern std::vector<std::string> CAM_NAMES;

extern int ROW;
extern int COL;
extern int FOCAL_LENGTH;
const int NUM_OF_CAM = 1;

extern int MAX_CNT;
extern int MIN_DIST;
extern int WINDOW_SIZE;
extern int FREQ;
extern int SHOW_TRACK;
extern int SHOW_UNDISTORTION;
extern int EQUALIZE;
extern int FISHEYE;
extern bool PUB_THIS_FRAME;
extern bool STEREO_TRACK;
extern double F_THRESHOLD;

void readParameters(ros::NodeHandle &n);

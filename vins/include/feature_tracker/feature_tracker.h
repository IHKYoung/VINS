#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "feature_parameters.h"
#include "utility/tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTracker
{
public:
    FeatureTracker();

    void readImage(const cv::Mat &_img, double _cur_time);

    void setMask();

    void addPoints();

    bool updateID(unsigned int i);

    void readIntrinsicParameter(const string &calib_file);

    void showUndistortion(const string &name);

    void rejectWithF();

    void undistortedPoints();

    cv::Mat               mask;
    cv::Mat               fisheye_mask;
    cv::Mat               prev_img, cur_img, forw_img; // 前一帧图像、当前帧图像（光流跟踪前一帧）、后一帧图像（光流跟踪后一帧）
    vector<cv::Point2f>   n_pts;
    vector<cv::Point2f>   prev_pts, cur_pts, forw_pts;
    vector<cv::Point2f>   prev_un_pts, cur_un_pts;
    vector<cv::Point2f>   pts_velocity; // 当前帧相对前一帧的特征点，沿xy方向的像素移动速度
    vector<int>           ids;          // 能够被跟踪到的特征点的id
    vector<int>           track_cnt;
    map<int, cv::Point2f> cur_un_pts_map;
    map<int, cv::Point2f> prev_un_pts_map;
    camodocal::CameraPtr  m_camera; // 相机模型
    double                cur_time;
    double                prev_time;

    static int n_id; // 特征点id，每检测到一个新的特征点，就将n_id作为该特征点的id，然后n_id加1
};

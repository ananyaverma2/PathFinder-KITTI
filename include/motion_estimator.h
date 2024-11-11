//
// Created by Ananya on 04/11/2024.
//

#ifndef MOTION_ESTIMATOR_H
#define MOTION_ESTIMATOR_H

#include <opencv2/opencv.hpp>
#include "dataset_handler.h"

class MotionEstimator {
public:

  MotionEstimator();
  void EstimateMotionUsingPnP(const std::vector<cv::DMatch>& matches,
    const std::vector<cv::KeyPoint>& left_keypoints,
    const std::vector<cv::KeyPoint>& right_keypoints,
    const DatasetHandler::CameraParameters& params,         // Intrinsics matrix of the left camera
    const cv::Mat& depth_image,           // Depth map of the reference image
    cv::Mat& rvec, cv::Mat& tvec);
};

#endif //MOTION_ESTIMATOR_H

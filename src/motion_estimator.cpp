//
// Created by Ananya on 04/11/2024.
//
#include "../include/motion_estimator.h"

#include <opencv2/opencv.hpp>

MotionEstimator::MotionEstimator() {};

void MotionEstimator::EstimateMotionUsingPnP(const std::vector<cv::DMatch>& matches,
  const std::vector<cv::KeyPoint>& left_keypoints,
  const std::vector<cv::KeyPoint>& right_keypoints,
  const DatasetHandler::CameraParameters& params,         // Intrinsics matrix of the left camera
  const cv::Mat& depth_image,           // Depth map of the reference image
  cv::Mat& rvec, cv::Mat& tvec) {

  std::vector<cv::Point3f> points3D;
  std::vector<cv::Point2f> points2D;

  for (const auto& match : matches) {
    cv::Point2f left_point = left_keypoints[match.queryIdx].pt;
    cv::Point2f right_point = right_keypoints[match.queryIdx].pt;

    // Depth at the location of pt0
    float d = depth_image.at<float>(cvRound(left_point.y), cvRound(left_point.x));
    if (d <= 0.0) continue;  // Skip invalid depth points

    // Compute 3D point (X, Y, Z) in camera coordinates for pt0
    float X = (left_point.x - params.cx) * d / params.fx;
    float Y = (left_point.y - params.cy) * d / params.fy;
    float Z = d;

    points3D.emplace_back(X, Y, Z);
    points2D.push_back(right_point);
  }

  bool success = cv::solvePnP(points3D, points2D,cv::Mat::eye(3, 3, CV_64F), cv::Mat(), // Assuming identity matrix for intrinsic matrix
    rvec, tvec,
    false, cv::SOLVEPNP_ITERATIVE
  );
  if (!success) {
    std::cerr << "Motion estimation failed." << std::endl;
  }
  else {
    std::cout << "Motion estimator works!" << std::endl;
  }
}
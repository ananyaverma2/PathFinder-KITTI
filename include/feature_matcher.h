//
// Created by Ananya on 04/11/2024.
//

#ifndef FEATURE_MATCHER_H
#define FEATURE_MATCHER_H

#include <opencv2/opencv.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/core.hpp"

#include "../include/dataset_handler.h"

class FeatureMatcher {
public:
    const float kThreshold = 0.7f;
    std::vector<cv::DMatch> inliers;

    FeatureMatcher();
    std::vector<cv::DMatch> MatchORBFeatures(cv::Mat& descriptor1, cv::Mat& descriptor2);
};

#endif //FEATURE_MATCHER_H

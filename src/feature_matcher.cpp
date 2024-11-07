//
// Created by Ananya on 04/11/2024.
//


#include "../include/dataset_handler.h"
#include "../include/feature_matcher.h"

FeatureMatcher::FeatureMatcher() {}

std::vector<cv::DMatch> FeatureMatcher::MatchORBFeatures(cv::Mat& descriptor1, cv::Mat& descriptor2) {

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> inliers;
    matcher->match(descriptor1, descriptor2, inliers);

    return inliers;

}

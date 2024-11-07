//
// Created by Ananya on 04/11/2024.
//

#include "../include/feature_extractor.h"
#include "../include/dataset_handler.h"

FeatureExtractor::FeatureExtractor() {}

void FeatureExtractor::ExtractORBFeatures(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {

    cv::Ptr<cv::ORB> orb_extractor =  cv::ORB::create();
    orb_extractor->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
}

//
// Created by Ananya on 04/11/2024.
//

#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H


#include <opencv2/opencv.hpp>

#include "../include/dataset_handler.h"

class FeatureExtractor {
public:
    FeatureExtractor();
    void ExtractORBFeatures(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
};

#endif //FEATURE_EXTRACTOR_H

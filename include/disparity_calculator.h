//
// Created by Ananya on 04/11/2024.
//

#ifndef DISPARITY_CALCULATOR_H
#define DISPARITY_CALCULATOR_H


#include <opencv2/opencv.hpp>

#include "../include/feature_extractor.h"

class DisparityCalculator {
public:
    DisparityCalculator();
    cv::Mat CalculateDisparity(cv::Mat& image_left, cv::Mat image_right);
private:
    const int kSadWindow = 6;
    const int kNumDisparities = kSadWindow*16;
    int block_size_ = 11;
};

#endif //DISPARITY_CALCULATOR_H

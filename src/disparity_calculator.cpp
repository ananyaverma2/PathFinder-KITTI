//
// Created by Ananya on 04/11/2024.
//

#include "../include/disparity_calculator.h"

DisparityCalculator::DisparityCalculator() {}

cv::Mat DisparityCalculator::CalculateDisparity(cv::Mat& image_left, cv::Mat image_right) {

    if (image_left.empty() || image_right.empty()) {
        std::cerr << "One or both input images are empty!" << std::endl;
        return cv::Mat();
    }

    cv::Mat disp;
    cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create();
    stereo->setNumDisparities(kNumDisparities);
    stereo->setBlockSize(block_size_);
    stereo->compute(image_left, image_right, disp);

    if (disp.empty()) {
        std::cerr << "Disparity computation failed." << std::endl;
        return cv::Mat();
    }
    cv::Mat image_disparity;
    disp.convertTo(image_disparity, CV_8U);

    return image_disparity;
}

//
// Created by Ananya on 04/11/2024.
//

#ifndef DATASET_HANDLER_H
#define DATASET_HANDLER_H

#pragma once

#include <string>
#include <iostream>

#include <opencv2/opencv.hpp>

#include <filesystem>
#include <vector>

class DatasetHandler {
public:
    DatasetHandler();
    void ReadImages();
    bool NextImages(cv::Mat& leftImage, cv::Mat& rightImage);
    int GetCurrentIndex() const;
private:
    std::vector<std::filesystem::path> left_images_;
    std::vector<std::filesystem::path> right_images_;
    int current_index_;
};

#endif //DATASET_HANDLER_H
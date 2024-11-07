//
// Created by Ananya on 04/11/2024.
//

#include "../include/dataset_handler.h"

DatasetHandler::DatasetHandler() {
    this->current_index_ = 0;
}

void DatasetHandler::ReadImages() {

    std::filesystem::path left_directory_path = "../data/dataset/sequences/test/image_0";
    std::filesystem::path right_directory_path = "../data/dataset/sequences/test/image_1";

    if (std::filesystem::exists(right_directory_path) && std::filesystem::exists(left_directory_path)) {
        for (auto& left_image: std::filesystem::directory_iterator(left_directory_path)) {

            std::cout << left_image.path() << std::endl;
            left_images_.push_back(left_image.path());
        }
        for (auto& right_image: std::filesystem::directory_iterator(right_directory_path)) {

            std::cout << right_image.path() << std::endl;
            right_images_.push_back(right_image.path());
        }
    }
    else {
        std::cout << "folder doesnt exist " << std::endl;
    }
}

bool DatasetHandler::NextImages(cv::Mat& leftImage, cv::Mat& rightImage) {

    if (current_index_ >= left_images_.size() || current_index_ >= right_images_.size()) {
        std::cout << "End of dataset reached." << std::endl;
        return false;
    }

    // Load images at currentIndex
    leftImage = cv::imread(left_images_[current_index_].string(), cv::IMREAD_GRAYSCALE);
    rightImage = cv::imread(right_images_[current_index_].string(), cv::IMREAD_GRAYSCALE);

    if (leftImage.empty() || rightImage.empty()) {
        std::cerr << "Failed to load images at index " << current_index_ << std::endl;
        return false;
    }

    current_index_++; // Move to the next pair for future calls
    return true;
}


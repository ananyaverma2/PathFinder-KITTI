//
// Created by Ananya on 04/11/2024.
//

#include "../include/dataset_handler.h"

#include <fstream>

DatasetHandler::DatasetHandler() {
    this->current_index_ = 0;
}

void DatasetHandler::ReadImages() {

    std::filesystem::path left_directory_path = "../data/dataset/sequences/01/image_0";
    std::filesystem::path right_directory_path = "../data/dataset/sequences/01/image_1";

    if (std::filesystem::exists(right_directory_path) && std::filesystem::exists(left_directory_path)) {
        for (auto& left_image: std::filesystem::directory_iterator(left_directory_path)) {
            left_images_.push_back(left_image.path());
        }
        for (auto& right_image: std::filesystem::directory_iterator(right_directory_path)) {
            right_images_.push_back(right_image.path());
        }
    }
    else {
        std::cout << "folder doesnt exist " << std::endl;
    }
}

bool DatasetHandler::NextImages(cv::Mat& leftImage, cv::Mat& rightImage) {

    if (current_index_ >= left_images_.size() || current_index_ >= right_images_.size()) {
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

DatasetHandler::CameraParameters DatasetHandler::GetCameraParameters(const std::string& file_path) {

    std::ifstream file(file_path);
    CameraParameters parameters = {0,0,0,0,0,};

    if (!file.is_open()) {
        std::cout << "calib.txt could not be read at " << file_path << std::endl;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string key;
        ss >> key;

        if (key == "P0:") {
            std::vector<double> values;
            double value;
            while (ss >> value) {
                values.push_back(value);
            }
            if (values.size() == 12) {
                cv::Mat projection_matrix_left = cv::Mat(values).reshape(1,3);

                cv::Mat camera_matrix_left, rotation_matrix_left, translation_vector_left;
                cv::decomposeProjectionMatrix(projection_matrix_left, camera_matrix_left, rotation_matrix_left, translation_vector_left);

                parameters.fx = camera_matrix_left.at<double>(0, 0);
                parameters.fy = camera_matrix_left.at<double>(1, 1);
                parameters.cx = camera_matrix_left.at<double>(0, 2);
                parameters.cy = camera_matrix_left.at<double>(1, 2);
            }
        }
        if (key == "P1:") {
            std::vector<double> values_right;
            double value_right;
            while (ss >> value_right) {
                values_right.push_back(value_right);
            }
            if (values_right.size() == 12) {
                cv::Mat projection_matrix_right = cv::Mat(values_right).reshape(1,3);

                cv::Mat camera_matrix_right, rotation_matrix_right, translation_vector_right;
                cv::decomposeProjectionMatrix(projection_matrix_right, camera_matrix_right, rotation_matrix_right, translation_vector_right);

                parameters.baseline = std::abs(translation_vector_right.at<double>(0) / parameters.fx);
            }
        }
    }
    file.close();
    return parameters;
}

void DatasetHandler::GetGroundTruth(std::vector<cv::Mat>& rotations, std::vector<cv::Mat>& translations) {

    std::filesystem::path poses_directory_path = "../data/dataset/poses/01.txt";

    std::ifstream file(poses_directory_path);

    if (!file.is_open()) {
        std::cout << "poses.txt could not be read at " << poses_directory_path << std::endl;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::vector<double> values(12); // To store 12 values for rotation and translation

        // Read the 12 values from each line
        for (int i = 0; i < 12; ++i) {
            ss >> values[i];
        }
        // Rotation matrix (3x3)
        cv::Mat rotation_matrix = cv::Mat(3, 3, CV_64F);
        int index = 0;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                rotation_matrix.at<double>(i, j) = values[index++];
            }
        }

        // Translation vector (3x1)
        cv::Mat translation_vector = cv::Mat(3, 1, CV_64F);
        for (int i = 0; i < 3; ++i) {
            translation_vector.at<double>(i, 0) = values[index++];
        }

        // Store the rotation matrix and translation vector
        rotations.push_back(rotation_matrix);
        translations.push_back(translation_vector);
    }
    file.close();
}

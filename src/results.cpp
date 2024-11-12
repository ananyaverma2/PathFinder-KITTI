//
// Created by Ananya on 04/11/2024.
//

#include "../include/results.h"


Results::Results() {};

void Results::SavePosesToFile(const std::vector<cv::Mat>& rotations,
                     const std::vector<cv::Mat>& translations,
                     const std::string& filename) {
    std::ofstream file(filename);

    // Check if the file is open
    if (!file.is_open()) {
        std::cerr << "Error opening file for writing!" << std::endl;
        return;
    }

    // Iterate through the poses
    for (size_t i = 0; i < rotations.size(); ++i) {
        // Get the rotation matrix from the rotation vector
        cv::Mat rotation_matrix;
        cv::Rodrigues(rotations[i], rotation_matrix); // Converts rvec to rotation matrix

        // Extract translation vector
        cv::Mat translation = translations[i];

        // Write the pose ID (99, 100, etc.)
        file << i << " ";

        // Write the rotation matrix (3x3) and translation (3x1) in row-major order
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                file << rotation_matrix.at<double>(row, col) << " ";
            }
        }

        // Write the translation vector (T03, T13, T23)
        for (int row = 0; row < 3; ++row) {
            file << translation.at<double>(row) << " ";
        }

        // End of line for the current pose
        file << std::endl;
    }

    file.close();
}
#include <iostream>
#include "include/dataset_handler.h"
#include "include/feature_extractor.h"
#include "include/feature_matcher.h"
#include "include/disparity_calculator.h"
#include "include/motion_estimator.h"
#include "include/results.h"

int main() {
    DatasetHandler dataset;
    FeatureExtractor featureExtractor;
    FeatureMatcher featureMatcher;
    DisparityCalculator disparityCalculator;
    MotionEstimator motionEstimator;
    Results saveResults;

    dataset.ReadImages();
    std::string file_path = "../data/dataset/sequences/01/calib.txt";
    std::vector<cv::Mat> rotations,  translations;
    dataset.GetGroundTruth(rotations, translations);

    DatasetHandler::CameraParameters params =  dataset.GetCameraParameters(file_path);

    cv::Mat left_image, right_image;

    // Vector to accumulate estimated rotations and translations for all frames
    std::vector<cv::Mat> all_rotations, all_translations;
    cv::Mat cumulative_translations = cv::Mat::zeros(3, 1, CV_64F);


    while (dataset.NextImages(left_image, right_image)) {
        std::vector<cv::KeyPoint> left_keypoints, right_keypoints;
        cv::Mat left_descriptors, right_descriptors;

        featureExtractor.ExtractORBFeatures(left_image, left_keypoints, left_descriptors);
        featureExtractor.ExtractORBFeatures(right_image, right_keypoints, right_descriptors);

        cv::Mat left_image_with_features, right_image_with_features;
        cv::drawKeypoints(left_image, left_keypoints, left_image_with_features, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::drawKeypoints(right_image, right_keypoints, right_image_with_features, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        std::vector<cv::DMatch> matches = featureMatcher.MatchORBFeatures(left_descriptors, right_descriptors);
        cv::Mat matched_images;
        cv::drawMatches(left_image, left_keypoints, right_image, right_keypoints, matches, matched_images);

        cv::Mat disparity_image;
        disparity_image = disparityCalculator.CalculateDisparity(left_image, right_image);
        disparity_image.convertTo(disparity_image, CV_32F); // scale factor might vary, adjust if needed

        cv::Mat depth_map(disparity_image.size(), CV_32F);
        for (int i=0; i<disparity_image.rows; i++) {
            for (int j=0; j<disparity_image.cols; j++) {
                if (disparity_image.at<float>(i,j) >= 1.0) {
                    depth_map.at<float>(i, j) = (params.fx * params.baseline) / disparity_image.at<float>(i,j);
                    //std::cout << (params.fx * params.baseline) << ";" << disparity_image.at<float>(i,j) << std::endl;
                }
                else {
                    depth_map.at<float>(i, j) = 0;
                }
            }
        }
        cv::Mat estimated_rotations, estimated_translations;
        motionEstimator.EstimateMotionUsingPnP(matches, left_keypoints, right_keypoints, params, depth_map, estimated_rotations, estimated_translations);

        // Accumulate the estimated translation and rotation
        all_rotations.push_back(estimated_rotations.clone());  // Store rotation matrix
        cumulative_translations += estimated_translations;  // Accumulate translation
        all_translations.push_back(cumulative_translations.clone());  // Store accumulated translation

    }

    std::cout << "Saving results..." << std::endl;
    // Save all poses to file at once
    saveResults.SavePosesToFile(all_rotations, all_translations, "../results/poses.txt");

    return 0;
}

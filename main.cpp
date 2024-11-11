#include <iostream>
#include "include/dataset_handler.h"
#include "include/feature_extractor.h"
#include "include/feature_matcher.h"
#include "include/disparity_calculator.h"
#include "include/motion_estimator.h"

int main() {
    DatasetHandler dataset;
    FeatureExtractor featureExtractor;
    FeatureMatcher featureMatcher;
    DisparityCalculator disparityCalculator;
    MotionEstimator motionEstimator;

    dataset.ReadImages();
    std::string file_path = "../data/dataset/sequences/test/calib.txt";

    DatasetHandler::CameraParameters params =  dataset.GetCameraParameters(file_path);

    cv::Mat left_image, right_image;

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
        cv::Mat rvec, tvec;
        motionEstimator.EstimateMotionUsingPnP(matches, left_keypoints, right_keypoints, params, depth_map, rvec, tvec);

        cv::imshow("matched image", matched_images);
        cv::waitKey(1);
        cv::imshow("disparity image", disparity_image);
        cv::waitKey(1);
        cv::Mat depthMapVis;
        cv::normalize(depth_map, depthMapVis, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::imshow("Depth Map", depthMapVis);
        cv::waitKey(1);
    }
    return 0;
}

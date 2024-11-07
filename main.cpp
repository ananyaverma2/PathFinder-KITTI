#include <iostream>
#include "include/dataset_handler.h"
#include "include/feature_extractor.h"
#include "include/feature_matcher.h"
#include "include/disparity_calculator.h"

int main() {
    DatasetHandler dataset;
    FeatureExtractor featureExtractor;
    FeatureMatcher featureMatcher;
    DisparityCalculator disparityCalculator;

    dataset.ReadImages();

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

        cv::imshow("matched image", matched_images);
        cv::waitKey(1);
        cv::imshow("disparity image", disparity_image);
        cv::waitKey(1);
    }
    return 0;
}

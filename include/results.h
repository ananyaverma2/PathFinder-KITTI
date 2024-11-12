#ifndef PLOTTER_H
#define PLOTTER_H

#include <fstream>
#include <opencv2/opencv.hpp>

class Results {
public:
  Results();
  void SavePosesToFile(const std::vector<cv::Mat>& rotations,
                     const std::vector<cv::Mat>& translations,
                     const std::string& filename);

};

#endif // PLOTTER_H

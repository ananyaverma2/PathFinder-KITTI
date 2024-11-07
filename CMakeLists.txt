cmake_minimum_required(VERSION 3.28)
project(visual_odometry)

set(CMAKE_CXX_STANDARD 17)

add_executable(visual_odometry main.cpp
        include/dataset_handler.h
        include/feature_extractor.h
        include/feature_matcher.h
        include/disparity_calculator.h
        include/motion_estimator.h
        include/plotter.h
        src/dataset_handler.cpp
        src/feature_extractor.cpp
        src/feature_matcher.cpp
        src/disparity_calculator.cpp
        src/motion_estimator.cpp
        src/plotter.cpp)

set(OpenCV_DIR "C:/opencv/build/install/x64/mingw/lib")
find_package(OpenCV 4 REQUIRED)


# Link against OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(visual_odometry ${OpenCV_LIBS})

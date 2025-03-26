cmake_minimum_required(VERSION 3.18)
project(esim)

## Compile as C++17, supported in ROS Noetic and newer
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

## 1) Find catkin and other ROS dependencies
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  cv_bridge
  dv_ros_msgs     # needed because node_event.cpp uses dv_ros_msgs/Event.h
)

## 2) (Optional) If libtorch is NOT in a default system location
# list(APPEND CMAKE_PREFIX_PATH "/home/sas/libtorch")

if(DEFINED ENV{LIBTORCH})
  list(APPEND CMAKE_PREFIX_PATH "$ENV{LIBTORCH}")
endif()

## 3) Find Torch
find_package(Torch REQUIRED)

## 4) Also find OpenCV (because of cv_bridge)
find_package(OpenCV REQUIRED)

## 4.5) Find Eigen3 for #include <Eigen/Dense>
find_package(Eigen3 REQUIRED NO_MODULE)

## 5) Declare this catkin package
catkin_package(
  CATKIN_DEPENDS roscpp std_msgs cv_bridge dv_ros_msgs
  # You don't list Eigen3 here since it's not a catkin package
)

## 6) Include directories
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${Torch_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}  # <--- Add Eigen3 include path
)

## 7) Enable CUDA if you have .cu files
enable_language(CUDA)

## 8) Build library from .cpp & .cu
add_library(esim_lib
  src/esim_cuda.cpp
  src/esim_cuda_kernel.cu
  src/node_event.cpp
)

## 9) Link your library (Torch, OpenCV, Python if needed)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
target_link_libraries(esim_lib
  ${catkin_LIBRARIES}
  ${TORCH_LIBRARIES}
  ${OpenCV_LIBS}
  Python3::Python
)

## 10) Build node_event executable
add_executable(node_event src/node_event.cpp)

target_link_libraries(node_event
  esim_lib
  ${catkin_LIBRARIES}
  ${TORCH_LIBRARIES}
  ${OpenCV_LIBS}
)

## 11) Make sure messages are generated first (important for dv_ros_msgs)
add_dependencies(node_event
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

## 12) Install rules (typical for catkin)
install(TARGETS esim_lib node_event
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

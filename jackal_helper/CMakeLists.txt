cmake_minimum_required(VERSION 3.0.2)
project(jackal_helper)

## Compile as C++17, supported in ROS Noetic and newer
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  message_generation
  geometry_msgs
  gazebo_msgs
  message_runtime
)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS system filesystem regex)
find_package(Protobuf REQUIRED)
include(FindPkgConfig)
if (PKG_CONFIG_FOUND)
  pkg_check_modules(GAZEBO REQUIRED gazebo)
endif()

catkin_package(
  CATKIN_DEPENDS roscpp std_msgs gazebo_msgs message_runtime
  DEPENDS Boost Protobuf GAZEBO
)

## Include directories
include_directories(
  /usr/include/x86_64-linux-gnu
  ${catkin_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${PROTOBUF_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/plugins
)

## Link directories
link_directories(${GAZEBO_LIBRARY_DIRS} ${CMAKE_SOURCE_DIR}/plugins/build)

## Find all shared object files in plugins/build
file(GLOB PLUGIN_LIBS "${CMAKE_SOURCE_DIR}/plugins/build/*.so")

## Add executable
add_executable(collision_publisher_node src/collision_publisher.cpp)

## Link the plugin libraries with the executable
target_link_libraries(collision_publisher_node
  ${catkin_LIBRARIES}
  ${GAZEBO_LIBRARIES}
  ${Boost_LIBRARIES}
  ${PROTOBUF_LIBRARIES}
  pthread
  ${PLUGIN_LIBS}  # Link all found .so files
)

## Install targets
install(TARGETS collision_publisher_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Set the GAZEBO_PLUGIN_PATH
set(GAZEBO_PLUGIN_PATH ${CMAKE_SOURCE_DIR}/plugins/build:${GAZEBO_PLUGIN_PATH})

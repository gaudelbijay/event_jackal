cmake_minimum_required(VERSION 3.0.2)
project(dv_ros)


find_package(catkin REQUIRED COMPONENTS
  dv_ros_msgs
  roscpp
  std_msgs
)

catkin_package(

)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)



add_executable(node src/node.cpp)
target_link_libraries(node ${catkin_LIBRARIES})

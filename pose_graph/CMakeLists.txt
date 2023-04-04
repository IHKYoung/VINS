cmake_minimum_required(VERSION 2.8.3)
project(pose_graph)

set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS "-std=c++14")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
        roscpp
        std_msgs
        nav_msgs
        camera_model
        cv_bridge
        roslib
        message_filters
        nodelet
        )

find_package(OpenCV REQUIRED)

# 定义自己编译的Ceres的库目录
set(Ceres_LIBS_DIR /usr/local/ceres2.0/lib)
# 定义CeresConfig.cmake所在文件目录
set(Ceres_DIR ${Ceres_LIBS_DIR}/cmake/Ceres)
find_package(Ceres REQUIRED)
message(STATUS "Found Ceres: ${CERES_FOUND}")
message(STATUS "    Ceres Include Dirs: ${CERES_INCLUDE_DIRS}")
message(STATUS "    Ceres Libs: ${CERES_LIBRARIES}")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)

include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

include_directories(
        ${catkin_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${PROJECT_SOURCE_DIR}/include
)

catkin_package()

add_library(pose_graph_lib
        src/pose_graph/pose_graph.cpp
        src/keyframe/keyframe.cpp
        src/utility/CameraPoseVisualization.cpp
        include/ThirdParty/DBoW/BowVector.cpp
        include/ThirdParty/DBoW/FBrief.cpp
        include/ThirdParty/DBoW/FeatureVector.cpp
        include/ThirdParty/DBoW/QueryResults.cpp
        include/ThirdParty/DBoW/ScoringObject.cpp
        include/ThirdParty/DUtils/Random.cpp
        include/ThirdParty/DUtils/Timestamp.cpp
        include/ThirdParty/DVision/BRIEF.cpp
        include/ThirdParty/VocabularyBinary.cpp
        )
target_link_libraries(pose_graph_lib ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})
#message("catkin_lib  ${catkin_LIBRARIES}")

add_library(pose_graph_nodelet src/pose_graph_nodelet.cpp)
target_link_libraries(pose_graph_nodelet pose_graph_lib)
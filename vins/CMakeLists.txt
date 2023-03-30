cmake_minimum_required(VERSION 2.8.3)
project(vins)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

add_compile_definitions(PRINT_DEBUG=0)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    geometry_msgs
    nav_msgs
    tf
    cv_bridge
    camera_model
    roslib
    )

find_package(OpenCV 4 REQUIRED)
# message(WARNING "OpenCV_VERSION: ${OpenCV_VERSION}")

# 定义自己编译的Ceres的库目录
set(Ceres_LIBS_DIR /usr/local/ceres1.14/lib)
# 定义CeresConfig.cmake所在文件目录
set(Ceres_DIR ${Ceres_LIBS_DIR}/cmake/Ceres)
find_package(Ceres REQUIRED)
message(STATUS "Found Ceres: ${CERES_FOUND}")
message(STATUS "    Ceres Include Dirs: ${CERES_INCLUDE_DIRS}")
message(STATUS "    Ceres Libs: ${CERES_LIBRARIES}")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${CERES_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
)

catkin_package()

add_executable(vins_estimator
    src/estimator/estimator_node.cpp
    src/estimator/estimator.cpp
    src/feature_manager/feature_manager.cpp
    src/factor/pose_local_parameterization.cpp
    src/factor/projection_factor.cpp
    src/factor/projection_td_factor.cpp
    src/factor/marginalization_factor.cpp
    src/utility/estimator_parameters.cpp
    src/utility/utility.cpp
    src/utility/visualization.cpp
    src/utility/CameraPoseVisualization.cpp
    src/initial/solve_5pts.cpp
    src/initial/initial_aligment.cpp
    src/initial/initial_sfm.cpp
    src/initial/initial_ex_rotation.cpp
    )
target_link_libraries(vins_estimator ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES}) 

add_executable(feature_tracker
    src/feature_tracker/feature_tracker_node.cpp
    src/feature_tracker/feature_parameters.cpp
    src/feature_tracker/feature_tracker.cpp
    )
target_link_libraries(feature_tracker ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(feature_tracker_rgbd
    src/feature_tracker/feature_tracker_rgbd_node.cpp
    src/feature_tracker/feature_parameters.cpp
    src/feature_tracker/feature_tracker.cpp
    )
target_link_libraries(feature_tracker_rgbd ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(pose_graph
    src/pose_graph/pose_graph_node.cpp
    src/pose_graph/pose_graph.cpp
    src/pose_graph/keyframe.cpp
    src/pose_graph/PoseGraphCameraPoseVisualization.cpp
    ThirdParty/DBoW/BowVector.cpp
    ThirdParty/DBoW/FBrief.cpp
    ThirdParty/DBoW/FeatureVector.cpp
    ThirdParty/DBoW/QueryResults.cpp
    ThirdParty/DBoW/ScoringObject.cpp
    ThirdParty/DUtils/Random.cpp
    ThirdParty/DUtils/Timestamp.cpp
    ThirdParty/DVision/BRIEF.cpp
    ThirdParty/VocabularyBinary.cpp
    )
target_link_libraries(pose_graph ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES}) 
cmake_minimum_required(VERSION 3.5)

project(bundle_adjustment)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O3")
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -O3")

find_package(ament_cmake_auto REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(TBB REQUIRED)

ament_auto_find_build_dependencies()


ament_auto_add_library(bundle_adjustment_node SHARED
  include/bundle_adjustment_node.hpp
  include/view.hpp
  include/params.hpp
  include/vision.hpp
  include/utils.hpp
  include/types.hpp
  include/initializer.hpp
  include/maps.hpp
  include/local_handler.hpp
  include/bundle_adjustment.hpp
  include/optimization.hpp

  include/concurrency/concurrentqueue.h
  include/concurrency/blockingconcurrentqueue.h
  include/concurrency/lightweightsemaphore.h

  src/bundle_adjustment_node.cpp
  src/vision.cpp
  src/utils.cpp
  src/initializer.cpp
  src/maps.cpp
  src/local_handler.cpp
  src/optimization.cpp
  )
  
include_directories(
  include
  ${TBB_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  )
  
target_include_directories(bundle_adjustment_node
        SYSTEM
        PUBLIC
        
        )
target_link_libraries(bundle_adjustment_node
        ${OpenCV_LIBRARIES}
        Eigen3::Eigen
        Ceres::ceres
        ${TBB_LIBRARIES}
        ) 


rclcpp_components_register_node(bundle_adjustment_node
  PLUGIN "BundleAdjustment::BundleAdjustmentNode"
  EXECUTABLE bundle_adjustment_node_exe
)

# Ament Exporting
ament_auto_package(
    INSTALL_TO_SHARE
    launch
    param
)

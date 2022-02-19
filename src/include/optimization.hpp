#ifndef BUILD_SRC_INCLUDE_OPTIMIZATION_HPP_
#define BUILD_SRC_INCLUDE_OPTIMIZATION_HPP_

#include "bundle_adjustment.hpp"
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "types.hpp"
#include "vision.hpp"
#include "view.hpp"

namespace MonocularVO
{

class Optimization
{
public:
  using FrameSharedPtr = std::shared_ptr<Frame>;
  using Batch = std::vector<FrameSharedPtr>;

  struct LocalObservations
  {
   public:
    std::vector<cv::Vec6d> camera_poses;
    std::vector<std::vector<cv::Point2d>> points2D;
    std::vector<cv::Point3d> points3D;
  };


  static MonocularVO::Optimization::LocalObservations
  solve_local_ba(MonocularVO::LocalObservations& local_observations, const cv::Mat& K);


};

}

#endif  // BUILD_SRC_INCLUDE_OPTIMIZATION_HPP_

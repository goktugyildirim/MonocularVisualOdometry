//
// Created by goktug on 20.03.2022.
//
#ifndef BUILD_SRC_INCLUDE_INITIALIZER_HPP_
#define BUILD_SRC_INCLUDE_INITIALIZER_HPP_

#include "vision.hpp"
#include "params.hpp"
#include "frames.hpp"


namespace MonocularVO
{
class Initializer
{
public:
  using FrameSharedPtr = std::shared_ptr<Frame>;
  explicit Initializer(const MonocularVO::Params& params);

  bool try_init(FrameSharedPtr& ref_frame,
                FrameSharedPtr& curr_frame,
                std::vector<int>&vector_tracked_p3d_ids_local,
                std::vector<int>&vector_tracked_p3d_ids_global,
                std::vector<cv::Point3d>& vector_initial_p3d,
                const double& scale);



private:
  MonocularVO::Params m_params;





};

}
#endif  // BUILD_SRC_INCLUDE_INITIALIZER_HPP_


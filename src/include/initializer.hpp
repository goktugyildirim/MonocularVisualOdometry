//
// Created by goktug on 20.03.2022.
//
#ifndef BUILD_SRC_INCLUDE_INITIALIZER_HPP_
#define BUILD_SRC_INCLUDE_INITIALIZER_HPP_

#include "vision.hpp"
#include "params.hpp"


namespace MonocularVO
{
class Initializer
{
public:
  using FrameSharedPtr = std::shared_ptr<Frame>;
  explicit Initializer(const MonocularVO::Params& params);

  void try_init(const FrameSharedPtr& ref_frame,
                const FrameSharedPtr& curr_frame,
                std::vector<int>& tracked_p2d_ids);

private:
  MonocularVO::Params m_params;





};

}
#endif  // BUILD_SRC_INCLUDE_INITIALIZER_HPP_


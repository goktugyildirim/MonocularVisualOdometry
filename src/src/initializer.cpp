//
// Created by goktug on 20.03.2022.
//

#include "initializer.hpp"

namespace MonocularVO
{
Initializer::Initializer(const MonocularVO::Params& params)
: m_params(params)
{}

bool
Initializer::try_init(FrameSharedPtr &ref_frame,
                      FrameSharedPtr &curr_frame,
                      std::vector<int> &tracked_p2d_ids)
{
  std::cout << "Trying to init..." << std::endl;
  std::cout << "Count ref frame tracked point count before initialization:" <<
    tracked_p2d_ids.size() << std::endl;

  std::cout << "Count ref frame tracked point count before initialization:" <<
      curr_frame->keypoints_p2d.size() << std::endl;

  if (tracked_p2d_ids.size() != curr_frame->keypoints_p2d.size())
    std::cout << "Error." << std::endl;

  return false;
}

}

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
  if (tracked_p2d_ids.size() != curr_frame->keypoints_p2d.size())
    std::cout << "Error." << std::endl;

  if (curr_frame->is_ref_frame)
    return false;

  std::cout << "Doing initialization." << std::endl;
  Vision::detect_keypoints(m_params,
                           curr_frame->keypoints,
                           curr_frame->image_gray);
  Vision::desc_keypoints(m_params,
                         curr_frame->keypoints,
                         curr_frame->descriptors,
                         curr_frame->image_gray);
  std::cout << "Detected keypoint count: " << curr_frame->keypoints.size() << std::endl;
  return false;
}

}

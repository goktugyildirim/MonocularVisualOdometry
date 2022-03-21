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
                      std::vector<int> &tracked_p2d_ids,
                      const double& scale)
{
  if (tracked_p2d_ids.size() != curr_frame->keypoints_p2d.size())
    std::cout << "Error." << std::endl;

  if (curr_frame->is_ref_frame)
    return false;

  std::cout << "Doing initialization." << std::endl;

  std::vector<cv::Point2f> tracked_ref_keypoints;
  for (const int& tracked_id : tracked_p2d_ids)
    tracked_ref_keypoints.push_back(ref_frame->keypoints[tracked_id].pt);
  std::vector<cv::Point2f> tracked_curr_keypoints = curr_frame->keypoints_p2d;

  return false;
}

}

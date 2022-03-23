//
// Created by goktug on 22.03.2022.
//

#include "tracker.hpp"

namespace MonocularVO
{

Tracker::Tracker(const MonocularVO::Params& params)
: m_params(params),
  m_id_p2d(0),
  m_id_p3d(0),
  m_id_frame(0)
{

}


void
Tracker::track_p2ds(
  const std::vector<cv::Point2f>& curr_frame_kpts,
  const int& frame_id,
  int& id_p2d)
{
  m_id_frame = frame_id;
  for (int i=0; i<curr_frame_kpts.size(); i++)
  {
    map_frames_to_p2ds[m_id_frame].push_back(m_id_p2d);
    map_p2d_to_frames[m_id_p2d].push_back(curr_frame_kpts.at(i));
    m_id_p2d++;
  }
}





}

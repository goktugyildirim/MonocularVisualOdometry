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
  m_points2D.reserve(999999);
  m_points3D.reserve(999999);
  m_observations.reserve(999999);
}


void
Tracker::track_p2ds(
  const std::vector<cv::Point2f>& curr_frame_kpts,
  const bool& is_ref_frame,
  const int& frame_id,
  int& id_p2d)
{
  if (is_ref_frame)
  {

    // Define initial observed Point3Ds:

  }

  for (int i=0; i<curr_frame_kpts.size(); i++)
  {



    m_id_p2d++;
  }

  id_p2d = m_id_p2d;
}




}

//
// Created by goktug on 24.03.2022.
//
#include "local_map.hpp"

namespace MonocularVO
{

LocalMap::LocalMap(const Params &params)
: m_id_p2d(0),
  m_id_p3d(0),
  m_id_frame(0)
{

}

void LocalMap::build_observations(
  const int &id_frame, const bool &is_ref_frame,
  const std::vector<cv::Point2f> &kpts)
{
  m_id_frame = id_frame;

  for (const cv::Point2f& kpt: kpts)
  {
    // Push observation information to the graph:
    m_map_frame_to_p2d[m_id_frame][m_id_p2d] = kpt; // frames to point2D
    m_map_frame_to_p3d[m_id_frame][m_id_p3d] = cv::Point3d {0,0,0}; // frames to point3D
    m_map_p3d_to_id_frame[m_id_p3d][m_id_frame] = cv::Vec6d {0,0,0,0,0,0}; // Point3d to frames
    map_cam_pose_p3d_to_observations[{m_id_frame, m_id_p3d}] = m_id_p2d;

    ObservationSharedPtr observation = std::make_shared<MonocularVO::Observation>(
        m_id_frame, m_id_p2d , m_id_p3d,
        cv::Vec6d {0,0,0,0,0,0},cv::Mat(),kpt,
        cv::Point3d {0,0,0},
        true,is_ref_frame,false,
        false);
    m_map_observations[m_id_p2d] = observation;

    m_id_p2d++;
    if (is_ref_frame)
      m_id_p3d++;
  }
}


std::vector<int>
LocalMap::get_p2d_ids_of_frame(const int &id_frame)
{
  std::vector<int> p2d_ids;
  for (auto it = m_map_frame_to_p2d[id_frame].begin();
       it!=m_map_frame_to_p2d[id_frame].end(); it++)
  {
    p2d_ids.push_back(it->first);
  }
  return p2d_ids;
}

std::vector<cv::Point2f>
LocalMap::get_p2d_of_frame(const int &id_frame)
{
  std::vector<cv::Point2f> p2d;
  for (auto it = m_map_frame_to_p2d[id_frame].begin();
       it!=m_map_frame_to_p2d[id_frame].end(); it++)
  {
    p2d.push_back(it->second);
  }
  return p2d;
}


std::vector<int>
LocalMap::get_p3d_ids_of_frame(const int &id_frame)
{
  std::vector<int> p3d_ids;
  for (auto it = m_map_frame_to_p3d[id_frame].begin();
       it!=m_map_frame_to_p3d[id_frame].end(); it++)
  {
    p3d_ids.push_back(it->first);
  }
  return p3d_ids;
}


std::vector<cv::Point3d>
LocalMap::get_p3d_of_frame(const int &id_frame)
{
  std::vector<cv::Point3d> p3d;
  for (auto it = m_map_frame_to_p3d[id_frame].begin();
       it!=m_map_frame_to_p3d[id_frame].end(); it++)
  {
    p3d.push_back(it->second);
  }
  return p3d;
}


std::vector<int>
LocalMap::get_frame_ids_of_p3d(const int &id_p3d)
{
  std::vector<int> frame_ids;
  for(auto it=m_map_p3d_to_id_frame[id_p3d].begin();
       it!=m_map_p3d_to_id_frame[id_p3d].end(); it++)
  {
    std::cout << "Point id: " << id_p3d << " seen in frame: " << it->first << std::endl;
    frame_ids.push_back(it->first);
  }
  return frame_ids;
}


std::vector<cv::Vec6d>
LocalMap::get_camera_poses_of_p3d(const int &id_p3d)
{
  std::vector<cv::Vec6d> camera_poses;
  for(auto it=m_map_p3d_to_id_frame[id_p3d].begin();
       it!=m_map_p3d_to_id_frame[id_p3d].end(); it++)
  {
    camera_poses.push_back(it->second);
  }
  return camera_poses;
}


ObservationSharedPtr
LocalMap::get_observation(
  const std::pair<int, int> &pair_id_frame_and_p3d)
{
  int id_observation = map_cam_pose_p3d_to_observations[pair_id_frame_and_p3d];
  ObservationSharedPtr observation = m_map_observations[id_observation];
  std::cout << "Observation id: " << id_observation <<
    " | Frame id: " << observation->id_frame <<
    " | Point3D id: " << observation->id_point3d <<
    " | Point2D id: " << observation->id_point2d << std::endl;
  return observation;
}


}

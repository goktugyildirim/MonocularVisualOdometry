//
// Created by goktug on 24.03.2022.
//

#ifndef BUILD_SRC_INCLUDE_LOCAL_MAP_HPP_
#define BUILD_SRC_INCLUDE_LOCAL_MAP_HPP_
#include "params.hpp"
#include "types.hpp"
#include "vision.hpp"


namespace MonocularVO
{


class LocalMap
{
public:
  explicit LocalMap( const MonocularVO::Params& params);

  void
  build_observations(const int& id_frame,
                     const bool& is_ref_frame,
                     const std::vector<cv::Point2f>& kpts);


  std::vector<int> get_p2d_ids_of_frame(const int& id_frame);
  std::vector<cv::Point2f> get_p2d_of_frame(const int &id_frame);

  std::vector<int> get_p3d_ids_of_frame(const int& id_frame);
  std::vector<cv::Point3d> get_p3d_of_frame(const int &id_frame);

  std::vector<int> get_frame_ids_of_p3d(const int& id_p3d);
  std::vector<cv::Vec6d> get_camera_poses_of_p3d(const int& id_p3d);

  ObservationSharedPtr get_observation(const std::pair<int, int>& pair_id_frame_and_p3d);

  void remove_bad_observations(const int& id_frame,
                               const std::vector<uchar>& vec_is_ok);



  // Functions for camera pose:




private:
  MonocularVO::Params m_params;
  int m_id_p2d;
  int m_id_p3d;
  int m_id_frame;

  std::map<int, ObservationSharedPtr> m_map_observations; // *

  // Key: id_frame | Value: Map -> Key: id_p2d Value: p2d
  std::map<int, std::map<int, cv::Point2f>> m_map_frame_to_p2d; // *
  // Key: id_frame | Value: Map -> Key: id_p3d Value: p3d
  std::map<int, std::map<int, cv::Point3d>> m_map_frame_to_p3d;  // *
  // Key: id_p3d | Value: Map -> Key: id_frame Value: pose
  std::map<int, std::map<int, cv::Vec6d>> m_map_p3d_to_id_frame; // *
  // Key: first: id_frame ; second: id_p3d | Value: id_p2d
  std::map<std::pair<int, int>, int> map_cam_pose_p3d_to_observations;

};

typedef std::shared_ptr<LocalMap> LocalMapSharedPtr;


}

#endif // BUILD_SRC_INCLUDE_LOCAL_MAP_HPP_

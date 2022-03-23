//
// Created by goktug on 22.03.2022.
//

#ifndef BUILD_SRC_INCLUDE_TRACKER_HPP_
#define BUILD_SRC_INCLUDE_TRACKER_HPP_
#include "vision.hpp"
#include "frame.hpp"
#include "params.hpp"
#include "types.hpp"

#include <map>
#include <list>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

namespace MonocularVO
{
class Tracker
{
public:
  explicit Tracker(const MonocularVO::Params& params);

  void track_p2ds(const std::vector<cv::Point2f>& curr_frame_kpts,
                  const bool& is_ref_frame,
                  const int& frame_id,
                  int& id_p2d);

  void remove_bad_p2ds(std::vector<cv::Point2f>& curr_frame_kpts);

private:
  Params m_params;

  int m_id_p2d;
  int m_id_p3d;
  int m_id_frame;

  std::vector<cv::Point2f> m_points2D;
  std::vector<cv::Point3d> m_points3D;
  std::vector<ObservationSharedPtr> m_observations;

  // Mapping between Point3D to Observations:
  std::map<int, std::vector<int>> map_point3D_to_observation;

  // Mapping between Point2D to Frames
  std::map<int, std::vector<int>> m_map_p2d_to_frame;
  std::map<int, std::vector<int>> m_map_frame_to_p2d;
};

}
#endif // BUILD_SRC_INCLUDE_TRACKER_HPP_

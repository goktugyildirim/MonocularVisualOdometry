//
// Created by goktug on 22.03.2022.
//

#ifndef BUILD_SRC_INCLUDE_TRACKER_HPP_
#define BUILD_SRC_INCLUDE_TRACKER_HPP_
#include "vision.hpp"
#include "frame.hpp"
#include "params.hpp"

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
                  const int& frame_id,
                  int& id_p2d);

  void remove_bad_p2ds(std::vector<cv::Point2f>& curr_frame_kpts);

private:
  Params m_params;
  int m_id_p2d;
  int m_id_p3d;
  int m_id_frame;


  std::map<int, std::vector<cv::Point2f>> map_p2d_to_frames;
  std::map<int, std::vector<int>> map_frames_to_p2ds;

};

}
#endif // BUILD_SRC_INCLUDE_TRACKER_HPP_

#ifndef BUILD_SRC_INCLUDE_TYPES_HPP_
#define BUILD_SRC_INCLUDE_TYPES_HPP_

#include <iostream>
#include <mutex>
#include <frame.hpp>
#include <mutex>
#include <deque>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <Eigen/Dense>

namespace MonocularVO
{
  using FrameSharedPtr = std::shared_ptr<Frame>;
  struct Observation
  {
    bool is_ref_frame{};
    bool is_keyframe;
    int id_frame;
    cv::Vec6d pose;
    cv::Mat mat_pose_4x4;

    int id_point3d;
    cv::Point3d point3d;

    int id_point2d{};
    cv::Point2d point2d;

    bool is_optimized;
    bool is_initialized;

    Observation(
        const int &id_frame, const int& id_point2d, const int &id_point3d,
        const cv::Vec6d &pose,  const cv::Mat &mat_pose_4x4, const cv::Point2d &point2d,
        const cv::Point3d &point3d, const bool &is_keyframe, const bool& is_ref_frame,
        const bool& is_optimized, const bool& is_initialized)
    {
      this->id_point2d = id_point2d;
      this->id_frame = id_frame;
      this->pose = pose;
      this->mat_pose_4x4 = mat_pose_4x4;
      this->is_keyframe = is_keyframe;
      this->is_ref_frame;
      this->id_point3d = id_point3d;
      this->point3d = point3d;
      this->point2d = point2d;
      this->is_optimized = is_optimized;
      this->is_initialized = is_initialized;
    }
  };

  typedef std::shared_ptr<Observation> ObservationSharedPtr;


struct LocalObservations
{
 public:
  std::vector<cv::Vec6d> camera_poses;
  std::vector<std::vector<cv::Point2d>> points2D;
  std::vector<cv::Point3d> points3D;
};

  class MatchKeyFrame
  {
  public:
    using MatchKeyFrameSharedPtr = std::shared_ptr<MatchKeyFrame>;

    int old_frame_id;
    std::vector<cv::KeyPoint> old_keyframe_kpts;
    std::vector<int> old_kpt_ids;
    int new_frame_id;
    std::vector<cv::KeyPoint> new_keyframe_kpts;
    std::vector<int> new_kpt_ids;
    //cv::Mat img_concat;


    MatchKeyFrame(
        const int& old_frame_id,
        const std::vector<cv::KeyPoint>& old_keyframe_kpts,
        const std::vector<int>& old_kpt_ids,
        const int& new_frame_id,
        const std::vector<cv::KeyPoint>& new_keyframe_kpts,
        const std::vector<int>& new_kpt_ids
        //const cv::Mat& img_concat
        )
    {
      this->old_frame_id = old_frame_id;
      this->old_keyframe_kpts = old_keyframe_kpts;
      this->old_kpt_ids = old_kpt_ids;
      this->new_frame_id = new_frame_id;
      this->new_keyframe_kpts = new_keyframe_kpts;
      this->new_kpt_ids = new_kpt_ids;
      //this->img_concat = img_concat;
    }
  };

}

#endif  // BUILD_SRC_INCLUDE_TYPES_HPP_

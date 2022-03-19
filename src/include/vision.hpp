#ifndef BUILD_SRC_INCLUDE_VISION_HPP_
#define BUILD_SRC_INCLUDE_VISION_HPP_

#include <frames.hpp>
#include <iostream>
#include <thread>
#include <params.hpp>
#include <frame.hpp>
#include "types.hpp"
#include "utils.hpp"
#include "utils.hpp"
#include "frames.hpp"
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/sfm/fundamental.hpp>
#include <opencv2/sfm/projection.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

#include <algorithm>

namespace MonocularVO
{
class Vision
{
 public:

  using MapInitialSharedPtr = std::shared_ptr<Frames>;
  using MatchKeyFrameSharedPtr = std::shared_ptr<MatchKeyFrame>;




  static void
  adaptiveNonMaximalSuppresion(std::vector<cv::KeyPoint>& keypoints,
                               const int& numToKeep);

  static void
  detect_keypoints(const MonocularVO::Params& params,
                std::vector<cv::KeyPoint> &keypoints,
                cv::Mat& img);

  static void
  keypoints_modern(const MonocularVO::Params& params,
                   std::vector<cv::KeyPoint> &keypoints,
                cv::Mat &img, const std::string &detectorType);

  static void
  keypoints_shitomasi(std::vector<cv::KeyPoint> &keypoints,
                      cv::Mat &img,
                       const MonocularVO::Params& params);

  static void
  keypoints_harris(std::vector<cv::KeyPoint> &keypoints,
                      cv::Mat &img);


  static void
  desc_keypoints(const MonocularVO::Params& params,
                 std::vector<cv::KeyPoint> &keypoints,
                 cv::Mat& descriptors,
                 const cv::Mat& img);

  static void
  match_descriptors(std::vector<cv::KeyPoint> &kpts_source,
                    std::vector<cv::KeyPoint> &kpts_ref,
                    cv::Mat &desc_source,
                    cv::Mat &desc_ref,
                    std::vector<cv::DMatch> &matched_kpts,
                    const MonocularVO::Params& params);

  static void
  draw_keypoints(cv::Mat& image,
    const std::vector<cv::KeyPoint>& keypoints);

  static cv::Mat
  draw_matches(const MonocularVO::FrameSharedPtr& view1,
               const MonocularVO::FrameSharedPtr& view2);

  static void
  make_img_3_channel(cv::Mat& img);

  static cv::Mat
  get_F(MonocularVO::FrameSharedPtr& view1,
        MonocularVO::FrameSharedPtr& view2,
        const float& ransac_threshold,
        cv::Mat& inliers_F);

  static cv::Mat
  get_E(MonocularVO::FrameSharedPtr& view1,
        MonocularVO::FrameSharedPtr& view2,
        const float& ransac_threshold,
        cv::Mat& inliers_E,
        const cv::Mat& K);

  static void
  recover_pose(
        const MonocularVO::FrameSharedPtr& view1,
        const MonocularVO::FrameSharedPtr& view2,
        const cv::Mat& F,
        const cv::Mat& E,
        cv::Mat& R,
        cv::Mat& t,
        const cv::Mat& K);


  static void
  refine_matches(FrameSharedPtr& view1,
                 FrameSharedPtr& view2,
                 const cv::Mat& inliers_F,
                 const cv::Mat& inliers_E,
                 std::vector<int>& old_kpt_ids,
                 std::vector<int>& new_kpt_ids);


  static void
  extract_features(FrameSharedPtr& frame,
                   const MonocularVO::Params& params);



  static int
  track_features(
      const MonocularVO::Params& params,
      MapInitialSharedPtr & map,
      cv::Mat& R,
      cv::Mat& t);



  static MatchKeyFrameSharedPtr
  match_key_frames(
      const MonocularVO::Params& params,
      FrameSharedPtr& keyframe_old,
      FrameSharedPtr& keyframe_new);


  static cv::Mat
  visualize_feature_tracking(const MapInitialSharedPtr & map,
                             const bool& save_images,
                             const bool& draw_line);

  static double
  average_ang_px_displacement(const std::vector<cv::Point2f> &prev_frame,
                              const std::vector<cv::Point2f> &curr_frame,
                              const float& img_height, const float& img_width);

};

}


#endif  // BUILD_SRC_INCLUDE_VISION_HPP_

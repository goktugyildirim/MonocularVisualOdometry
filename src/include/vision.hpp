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
  extract_features(FrameSharedPtr& frame,
                   const MonocularVO::Params& params);

  static double
  average_ang_px_displacement(const std::vector<cv::Point2f> &prev_frame,
                              const std::vector<cv::Point2f> &curr_frame,
                              const float& img_height, const float& img_width);

  static void
  pose_estimation_2d2d(const std::vector<cv::Point2f> &kpts_prev_frame,
                       const std::vector<cv::Point2f> &kpts_curr_frame,
                       const MonocularVO::Params& params,
                       cv::Mat& R, cv::Mat& t);


  // ---------------- transformations ------------------
  static cv::Point2f pixel_2_cam_norm_plane(const cv::Point2f &p, const cv::Mat &K);
  static cv::Point3f pixel_2_cam(const cv::Point2f &p, const cv::Mat &K, double depth = 1);
  static cv::Point2f cam_2_pixel(const cv::Point3f &p, const cv::Mat &K);

};

}


#endif  // BUILD_SRC_INCLUDE_VISION_HPP_

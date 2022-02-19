#ifndef BUILD_SRC_INCLUDE_VIEW_HPP_
#define BUILD_SRC_INCLUDE_VIEW_HPP_

#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/affine.hpp>

namespace BundleAdjustment {
class Frame
{
 public:

  cv::Mat img_colored;
  cv::Mat image_gray;
  cv::Mat image_gray_with_kpts;
  int view_id{};
  int height{};
  int width{};
  std::vector<cv::KeyPoint> keypoints{};
  cv::Mat descriptors;
  bool is_key_frame = false;
  bool is_img_deleted = false;
  bool is_feature_extracted = false;

  // Each transform information will be updated at the same time.
  cv::Vec3d tf_rvec_world_to_cam;
  cv::Mat tf_R_world_to_cam;
  cv::Mat tf_t_world_to_cam;
  cv::Vec6d pose_cam_6dof_world_to_camera;

  std::vector<cv::Point2f> keypoints_pt{};
  std::vector<cv::Vec3d> points3D;

  void
  set_key_frame()
  {
    this->is_key_frame = true;
    cv::putText(image_gray_with_kpts,
                "Key Frame",
                cv::Point(75, 500),
                cv::FONT_HERSHEY_DUPLEX,
                3, CV_RGB(0, 0, 255), 4);
  }

  void
  void_set_odom_pose(const cv::Mat& R,
                     const cv::Mat& t)
  {

  }

 private:

};

}

#endif  // BUILD_SRC_INCLUDE_VIEW_HPP_

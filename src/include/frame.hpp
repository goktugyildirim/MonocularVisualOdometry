#ifndef BUILD_SRC_INCLUDE_FRAME_HPP_
#define BUILD_SRC_INCLUDE_FRAME_HPP_

#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/affine.hpp>

namespace MonocularVO {
class Frame
{
 public:

  cv::Mat img_colored;
  cv::Mat image_gray;
  cv::Mat image_gray_with_kpts;
  int frame_id{};
  int height{};
  int width{};
  std::vector<cv::KeyPoint> keypoints{};
  cv::Mat descriptors;
  cv::Mat descriptors_tracked;

  bool is_ref_frame = false;

  bool is_keyframe = false;
  bool is_img_deleted = false;
  bool is_feature_extracted = false;

  // Each transform information will be updated at the same time.
  cv::Vec3d tf_rvec_world_to_cam;
  cv::Mat tf_R_world_to_cam;
  cv::Mat tf_t_world_to_cam;
  cv::Vec6d pose_cam_6dof_world_to_camera;

  std::vector<cv::Point2f> keypoints_p2d{};
  std::vector<cv::Vec3d> points3D;

  void
  set_ref_frame()
  {
    this->is_ref_frame = true;
    this->is_keyframe = true;
    cv::putText(image_gray_with_kpts,
                "Reference Frame",
                cv::Point(75, 500),
                cv::FONT_HERSHEY_DUPLEX,
                2, CV_RGB(0, 0, 255), 3);

    cv::putText(img_colored,
                "Reference Frame",
                cv::Point(75, 500),
                cv::FONT_HERSHEY_DUPLEX,
                2, CV_RGB(0, 0, 255), 3);
  }

  void
  set_key_frame()
  {
    this->is_keyframe = true;
    cv::putText(image_gray_with_kpts,
                "Key Frame",
                cv::Point(75, 500),
                cv::FONT_HERSHEY_DUPLEX,
                3, CV_RGB(0, 0, 255), 4);
  }



 private:

};

}

#endif // BUILD_SRC_INCLUDE_FRAME_HPP_

#ifndef BUILD_SRC_INCLUDE_UTILS_HPP_
#define BUILD_SRC_INCLUDE_UTILS_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv4/opencv2/calib3d.hpp>
#include <std_msgs/msg/header.hpp>
#include "types.hpp"

namespace MonocularVO
{
class Utils
{
 public:
  using ImageMsgT = sensor_msgs::msg::Image;
  using ImageMsgPtrT = sensor_msgs::msg::Image::SharedPtr;

  static void
  publish_image(
      const rclcpp::Publisher<ImageMsgT>::SharedPtr& pub_img,
      const cv::Mat & image,
      const rclcpp::Time & time,
      const std::string & frame_id);

  static cv::Mat
  get_pose(const cv::Mat& R,
           const cv::Mat& t,
           const float& scale);

  static cv::Point2d
  project_point(const cv::Mat& R, const cv::Mat& t,
                const cv::Mat& K, const cv::Point3d& point_3D);

  static void
  get_not_matched_kpt_ids(const FrameSharedPtr& frame,
                          const std::vector<cv::DMatch>& matches,
                          std::vector<int>& vector_not_matched_kpt_ids,
                          const bool& is_prev_frame);

  static void
  remove_lost_landmark_ids(const std::vector<int>& list_of_index,
                                            std::vector<int>& vector_to_remove_elements);

  static void
  print_vector_elements(const std::vector<int>& vector_to_print);

  static void
  print_keypoints_with_indexes(const std::vector<cv::KeyPoint>& vector_to_print);

  static void
  print_keypoints_p2d_with_indexes(const std::vector<cv::Point2f>& vector_to_print);

  static void
  update_curr_frame_descriptor(
      const cv::Mat &matrix, const std::vector<cv::DMatch> &matches,
      cv::Mat &matrix_without_row);


};
}


#endif  // BUILD_SRC_INCLUDE_UTILS_HPP_

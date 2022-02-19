#ifndef BUILD_SRC_INCLUDE_UTILS_HPP_
#define BUILD_SRC_INCLUDE_UTILS_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv4/opencv2/calib3d.hpp>
#include <std_msgs/msg/header.hpp>
#include "types.hpp"

namespace BundleAdjustment
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


};
}


#endif  // BUILD_SRC_INCLUDE_UTILS_HPP_

#include <rclcpp/rclcpp.hpp>

#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "view.hpp"
#include "utils.hpp"
#include "types.hpp"
#include "monocular_vo_handler.hpp"

#include <string>
#include <memory>
#include <vector>
#include <queue>

namespace MonocularVO
{

class MonocularVONode : public rclcpp::Node
{
public:
  using FameSharedPtr = std::shared_ptr<Frame>;
  using LockFreeQueue = moodycamel::ConcurrentQueue<FameSharedPtr>;
  using ImageMsgT = sensor_msgs::msg::Image;
  using ImageMsgPtrT = sensor_msgs::msg::Image::SharedPtr;

  explicit MonocularVONode(
    const rclcpp::NodeOptions & node_options);

private:
  MonocularVO::Params params_;

  // Tracking:
  std::shared_ptr<LockFreeQueue> queue_frame_to_initialization_;
  void callback_view_tracked(
      const cv::Mat& img_concat);
  std::shared_ptr<Initializer> worker_initializer_;
  rclcpp::Publisher<ImageMsgT>::SharedPtr pub_match_view_;

  // Timer to provide DataFrame:
  rclcpp::TimerBase::SharedPtr timer_provide_data_frame_;
  void CallbackImageProvider();
  int view_id_;
  std::mutex door_;

};

}  // namespace MonocularVO






#include <rclcpp/rclcpp.hpp>

#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "frame.hpp"
#include "utils.hpp"
#include "types.hpp"
#include "local_tracking_handler.hpp"

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
  MonocularVO::Params m_params;

  // Tracking:
  std::shared_ptr<LockFreeQueue> m_queue_frames_to_local_tracking;
  void callback_view_tracked(
      const cv::Mat& img_concat);
  std::shared_ptr<LocalTrackingHandler> m_worker_local_tracker;
  rclcpp::Publisher<ImageMsgT>::SharedPtr m_pub_match_view;

  // Timer to provide DataFrame:
  rclcpp::TimerBase::SharedPtr m_timer_provide_data_frame;
  void CallbackImageProvider();
  int m_frame_id;
  std::mutex m_mutex;

};

}  // namespace MonocularVO






#include <monocular_vo_node.hpp>

namespace MonocularVO
{

MonocularVONode::MonocularVONode(
  const rclcpp::NodeOptions &node_options)
  : Node("bundle_adjustment_node", node_options), m_frame_id(0), // 400
      m_params(true, // The fastest combination : FAST - BRIEF - use modern: true
   "FAST","BRIEF",
   "BruteForce-Hamming","SEL_KNN",
   1000,9999,99999999,210,
   // The most important parameters:
   150, 5,
   20, 3,0.8)
{
  // Local Tracking ::
  m_queue_frames_to_local_tracking = std::make_shared<LockFreeQueue>(9999999);
  MonocularVO::LocalTrackingHandler::TypeCallbackTrack
      callback_view_tracked =
      std::bind(
          &MonocularVONode::callback_view_tracked,
          this, std::placeholders::_1
      );
  m_worker_local_tracker = std::make_shared<LocalTrackingHandler>(m_params,
      callback_view_tracked);
  m_worker_local_tracker->start(m_queue_frames_to_local_tracking);
  // Publishers:
  m_pub_match_view = this->create_publisher<ImageMsgT>(
      "/image_match", 50);
  // eof Local Tracking

  int ms = 1;
  m_timer_provide_data_frame = this->create_wall_timer(
      std::chrono::milliseconds(ms),
      std::bind(&MonocularVONode::CallbackImageProvider,
                this));
}

void
MonocularVONode::CallbackImageProvider()
{
  if (m_frame_id <840)
  {
    std::string img_name = "/home/goktug/projects/MonocularVisualOdometry/src/images/" +std::to_string(m_frame_id) + ".jpg";
    //std::string img_name = "/home/goktug/projects/MonocularVO/src/images/img.jpeg";
    cv::Mat img = imread(
        img_name,
        cv::IMREAD_COLOR);

    if (m_frame_id == 0)
    std::cout << "Image x:" << img.cols << " y:"  << img.rows << std::endl;




    bool use_undistorted_img = false;

    if (use_undistorted_img)
    {
      cv::Mat img_u;
      cv::undistort(img, img_u, m_params.K, m_params.mat_dist_coeff);
      img = img_u;
    }
    cv::Mat img_gray;
    cv::Mat img_gray_with_kpts;
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img, img_gray_with_kpts, cv::COLOR_BGR2GRAY);

    cv::putText(img,
                std::to_string(m_frame_id),
                cv::Point(75, 400),
                cv::FONT_HERSHEY_DUPLEX,
                3, CV_RGB(0, 0, 255), 4);


    FameSharedPtr view(new Frame);
    view->img_colored = img;

    Vision::make_img_3_channel(img_gray_with_kpts);

    cv::putText(img_gray_with_kpts,
                std::to_string(m_frame_id),
                cv::Point(75, 400),
                cv::FONT_HERSHEY_DUPLEX,
                3, CV_RGB(0, 0, 255), 4);

    view->image_gray = img_gray;
    view->frame_id = m_frame_id;
    view->width = img.cols;
    view->height = img.rows;
    view->image_gray_with_kpts = img_gray_with_kpts;

    //  Enqueues one item, but only if enough memory is already allocated
    while (!m_queue_frames_to_local_tracking->try_enqueue(view)) {
      // spin until write a value
    }

  }
//  else
//    m_frame_id = 0;
  m_frame_id++;
}


void
MonocularVONode::callback_view_tracked(
    const cv::Mat& img_concat)
{
/*
  int y = 900;
  int x = 900;
  cv::Mat cv_img(y, x, CV_8UC(3));
  cv_img.setTo(cv::Scalar(255,255,255));
  Utils::publish_image(m_pub_match_view, cv_img,
                       this->get_clock()->now(), "world");
*/
  Utils::publish_image(m_pub_match_view, img_concat,
   this->get_clock()->now(), "world");
}


}  // namespace MonocularVO


RCLCPP_COMPONENTS_REGISTER_NODE(
        MonocularVO::MonocularVONode)

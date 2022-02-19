#include <monocular_vo_node.hpp>

namespace MonocularVO
{

MonocularVONode::MonocularVONode(
  const rclcpp::NodeOptions &node_options)
  : Node("bundle_adjustment_node", node_options),
    view_id_(0),
    params_(false, // The fastest combination : FAST - BRIEF - use modern: true
   "SHITOMASI","BRIEF",
   "BruteForce-Hamming","SEL_KNN",
   3000,999999,99999999,160,
   // The most important parameters:
   250, 25,
   20, 8,1)
{
  // Initialization :::
  queue_frame_to_initialization_ = std::make_shared<LockFreeQueue>(30);
  MonocularVO::Initializer::TypeCallbackTrack
      callback_view_tracked =
      std::bind(
          &MonocularVONode::callback_view_tracked,
          this, std::placeholders::_1
      );
  worker_initializer_ = std::make_shared<Initializer>(
      params_,
      callback_view_tracked);
  worker_initializer_->start(queue_frame_to_initialization_);
  // Publishers:
  pub_match_view_ = this->create_publisher<ImageMsgT>(
      "/image_match", 50);
  // eof Tracking

  timer_provide_data_frame_ = this->create_wall_timer(
      std::chrono::milliseconds(40),
      std::bind(&MonocularVONode::CallbackImageProvider,
                this));
}

void
MonocularVONode::CallbackImageProvider()
{
  if (view_id_<840)
  {
    std::string img_name = "/home/goktug/projects/MonocularVisualOdometry/src/images/" +std::to_string(view_id_) + ".jpg";
    //std::string img_name = "/home/goktug/projects/MonocularVO/src/images/img.jpeg";
    cv::Mat img = imread(
        img_name,
        cv::IMREAD_COLOR);

    if (view_id_ == 0)
    std::cout << "Image x:" << img.cols << " y:"  << img.rows << std::endl;


    FameSharedPtr view(new Frame);
    view->img_colored = img;

    bool use_undistorted_img = true;

    if (use_undistorted_img)
    {
      cv::Mat img_u;
      cv::undistort(img, img_u,
                    params_.K,
                    params_.mat_dist_coeff);
      img = img_u;
    }
    cv::Mat img_gray;
    cv::Mat img_gray_with_kpts;
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img, img_gray_with_kpts, cv::COLOR_BGR2GRAY);

    Vision::make_img_3_channel(img_gray_with_kpts);

    cv::putText(img_gray_with_kpts,
                std::to_string(view_id_),
                cv::Point(75, 400),
                cv::FONT_HERSHEY_DUPLEX,
                3, CV_RGB(0, 0, 255), 4);

    view->image_gray = img_gray;
    view->view_id = view_id_;
    view->width = img.cols;
    view->height = img.rows;
    view->image_gray_with_kpts = img_gray_with_kpts;

    //  Enqueues one item, but only if enough memory is already allocated
    while (!queue_frame_to_initialization_->try_enqueue(view)) {
      // spin until write a value
    }

  }else
    view_id_ = 0;

  view_id_++;

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
  Utils::publish_image(pub_match_view_, cv_img,
                       this->get_clock()->now(), "world");
*/
  Utils::publish_image(pub_match_view_, img_concat,
   this->get_clock()->now(), "world");
}


}  // namespace MonocularVO


RCLCPP_COMPONENTS_REGISTER_NODE(
        MonocularVO::MonocularVONode)

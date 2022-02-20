#include "monocular_vo_handler.hpp"

namespace MonocularVO
{

MonocularVOHandler::MonocularVOHandler(
const MonocularVO::Params& params,
TypeCallbackTrack &callback_view_tracked)
: provide_{std::move(callback_view_tracked)},
  keep_visual_odometry_(true),
  params_(params)
{
    map_ = std::make_shared<Map>(params);
  // Local Handler Stuff:
  // * Local handler solves local BA problem
  // Batch :: [last_KF, N x KF,  curr_KF] --> Local Bundle Adjustment design
  queue_batch_to_local_handler_ = std::make_shared<LockFreeQueueBatch>(30);
  worker_local_handler_ = std::make_shared<LocalHandler>(params);
  worker_local_handler_->start(queue_batch_to_local_handler_);
}

void
MonocularVOHandler::start(std::shared_ptr<LockFreeQueue> &queue_view_to_initialization)
{
    future_worker_vo_handler_ = std::async(std::launch::async,
                                           &MonocularVOHandler::do_monocular_vo,
                                           this,
                                           std::ref(queue_view_to_initialization));
}

MonocularVOHandler::~MonocularVOHandler() {
  std::cout << "Shutdown MonocularVOHandlerHandler." << std::endl;
  future_worker_vo_handler_.get();
}

void
MonocularVOHandler::stop(){ keep_visual_odometry_ = false;}

void
MonocularVOHandler::do_monocular_vo(
    std::shared_ptr<LockFreeQueue> &queue_view_to_tracking)
{
  cv::Mat R_curr = cv::Mat::eye(3,3, 6);
  cv::Mat t_curr = cv::Mat::zeros(3,1,6);
  double scale = 1;
  int count_local_landmark;
  bool need_initialize = true;

  while (keep_visual_odometry_)
  {
    // Take frame from queue
    FrameSharedPtr frame(new Frame);
    while (!queue_view_to_tracking->try_dequeue(frame)) {}

    if (frame->view_id == 0)  // First frame is the key-frame:
    {
      frame->set_key_frame();
      Vision::extract_features(frame, params_);
      map_->push_frame(frame);
    }
    else // Not first frame
    {
      map_->push_frame(frame);
      cv::Mat R,t;
        count_local_landmark = Vision::track_features(
          params_, map_, R, t);

      // Scale can ben given from speed odometer or IMU | x = x0 + V*dt
       R_curr = R*R_curr;  t_curr = t_curr + scale*(R_curr*t);
       //std::cout << "Odometry translation: " << t_curr.t() << std::endl;

      provide_(Vision::visualize_feature_tracking(
          map_, false, false));

      // Condition check for selecting key-frame:
      // Condition 1:
      // You must do_monocular_vo at least 50 features
      // throughout the last 'max_frame_count_to_key_frame' frames!
      int diff_last_keyframe = (
          map_->get_curr_frame()->
              view_id - map_->get_last_key_frame()->view_id);
      bool condition1 = diff_last_keyframe > params_.max_frame_count_to_key_frame
                        and count_local_landmark > 50;
      // Condition 2:
      double curr_displacement = Vision::average_ang_px_displacement(
          map_->get_last_key_frame()->keypoints_pt,
          map_->get_curr_frame()->keypoints_pt,
          frame->width, frame->height);
      //std::cout << "Curr angular px displacement is " << int(curr_displacement) << " degree." << std::endl;
      bool condition2 = curr_displacement > params_.max_angular_px_disp;

      if(condition1 or condition2)
      {
        frame->set_key_frame();
        Batch batch;

        if (condition1)
        {
          std::cout << "\nMax frame count to be a key-frame is exceed." << std::endl;
          batch = map_->build_batch_1();
        }

        if (condition2)
        {
          std::cout << "\nAngular pixel displacement is exceed. View id: " <<
          frame->view_id  << " | "<< int(curr_displacement) <<
          " degree" <<  std::endl;
          batch = map_->build_batch_2();
        }

        std::cout << "Count current tracked feature from the last key-frame (keypoints_pt): "
                  << map_->count_local_landmark_ << std::endl;
        // Send local handler no need feature extraction:
        //map_->print_frames_info();
        try_send_batch_to_local_handler(batch);

      }

      // Condition 3:
      else if (count_local_landmark < params_.count_min_tracked)
      {
        std::cout << "\nFeature extraction required." << std::endl;
        frame->set_key_frame();
        Batch batch = map_->build_batch_2();

        // Send global handler

        Vision::extract_features(frame, params_);
        map_->delete_past_images();
        std::cout << "\n##########################################"
                     "##############################################" << std::endl;
      }
      }
    //map_->print_frames_info();
    } // eof view_id > 0

}


void
MonocularVOHandler::try_send_batch_to_local_handler(Batch& batch)
{
  Batch batch_copy;

  for (int i=0; i<batch.size(); i++)
  {
    FrameSharedPtr frame = batch.at(i);

    if (!frame->is_img_deleted and
    // Don't release images of the current frame:
    frame->view_id != map_->get_curr_frame()->view_id)
    {
      frame->img_colored.release();
      frame->image_gray_with_kpts.release();
      frame->image_gray.release();
      frame->is_img_deleted = true;
    }
    FrameSharedPtr frame_copy = std::make_shared<Frame>(*frame);
    batch_copy.push_back(frame_copy);
  }

  while (!queue_batch_to_local_handler_->try_enqueue(batch_copy)) {
    // spin until write a value
  }
}





} // end MonocularVO
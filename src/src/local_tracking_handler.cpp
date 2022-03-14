#include "local_tracking_handler.hpp"

#include <memory>

namespace MonocularVO
{

LocalTrackingHandler::LocalTrackingHandler(
const MonocularVO::Params& params,
TypeCallbackTrack &callback_view_tracked)
: send_to_ros_interface{std::move(callback_view_tracked)},
      m_keep_tracking(true), m_params(params)
{
  // Local Handler Stuff:
  // * Local handler solves local BA problem
  // Batch :: [last_KF, N x KF,  curr_KF] --> Local Bundle Adjustment design
  m_queue_batch_to_local_handler = std::make_shared<LockFreeQueueBatch>(30);
  m_worker_local_handler = std::make_shared<LocalHandler>(params);
  m_worker_local_handler->start(m_queue_batch_to_local_handler);
}

void LocalTrackingHandler::start(std::shared_ptr<LockFreeQueue> &queue_view_to_initialization)
{
  m_future_worker_local_tracking_handler = std::async(std::launch::async, &LocalTrackingHandler::track,
                                           this,
                                           std::ref(queue_view_to_initialization));
}

LocalTrackingHandler::~LocalTrackingHandler() {
  std::cout << "Shutdown MonocularVOHandlerHandler." << std::endl;
  m_future_worker_local_tracking_handler.get();
}

void LocalTrackingHandler::stop(){ m_keep_tracking = false;}

void LocalTrackingHandler::track(
    std::shared_ptr<LockFreeQueue> &queue_view_to_tracking)
{
  while (m_keep_tracking)
  {
    // Take curr_frame from queue
    FrameSharedPtr curr_frame(new Frame);
    while (!queue_view_to_tracking->try_dequeue(curr_frame)) {}



    std::cout << "curr_frame is  r" << std::endl;

/*    if (curr_frame->view_id == 0)  // First curr_frame is the key-curr_frame:
    {
      curr_frame->set_key_frame();
      Vision::extract_features(curr_frame, m_params);
      map_->push_frame(curr_frame);
    }
    else // Not first curr_frame
    {
      map_->push_frame(curr_frame);
      cv::Mat R,t;
        count_local_landmark = Vision::track_features(
          m_params, map_, R, t);

      // Scale can ben given from speed odometer or IMU | x = x0 + V*dt
       R_curr = R*R_curr;  t_curr = t_curr + scale*(R_curr*t);
       //std::cout << "Odometry translation: " << t_curr.t() << std::endl;

      send_to_ros_interface(Vision::visualize_feature_tracking(
          map_, false, false));

      // Condition check for selecting key-curr_frame:
      // Condition 1:
      // You must track at least 50 features
      // throughout the last 'max_frame_count_to_key_frame' frames!
      int diff_last_keyframe = (
          map_->get_curr_frame()->
              view_id - map_->get_last_key_frame()->view_id);
      bool condition1 = diff_last_keyframe > m_params.max_frame_count_to_key_frame
                        and count_local_landmark > 50;
      // Condition 2:
      double curr_displacement = Vision::average_ang_px_displacement(
          map_->get_last_key_frame()->keypoints_pt,
          map_->get_curr_frame()->keypoints_pt,
          curr_frame->width, curr_frame->height);
      //std::cout << "Curr angular px displacement is " << int(curr_displacement) << " degree." << std::endl;
      bool condition2 = curr_displacement > m_params.max_angular_px_disp;

      if(condition1 or condition2)
      {
        curr_frame->set_key_frame();
        Batch batch;

        if (condition1)
        {
          std::cout << "\nMax curr_frame count to be a key-curr_frame is exceed." << std::endl;
          batch = map_->build_batch_1();
        }

        if (condition2)
        {
          std::cout << "\nAngular pixel displacement is exceed. View id: " <<
          curr_frame->view_id  << " | "<< int(curr_displacement) <<
          " degree" <<  std::endl;
          batch = map_->build_batch_2();
        }

        std::cout << "Count current tracked feature from the last key-curr_frame (keypoints_pt): "
                  << map_->count_local_landmark_ << std::endl;
        // Send local handler no need feature extraction:
        //map_->print_frames_info();
        try_send_batch_to_local_handler(batch);

      }


      // Solve global BA
      // Condition 3:
      else if (count_local_landmark < m_params.count_min_tracked)
      {
        std::cout << "\nFeature extraction required." << std::endl;
        curr_frame->set_key_frame();
        Batch batch = map_->build_batch_2();

        // Send global handler

        Vision::extract_features(curr_frame, m_params);
        map_->delete_past_images();
        std::cout << "\n##########################################"
                     "##############################################" << std::endl;
      }
      }*/
    //map_->print_frames_info();
    } // eof view_id > 0

}

/*

void LocalTrackingHandler::try_send_batch_to_local_handler(Batch& batch)
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

  while (!m_queue_batch_to_local_handler->try_enqueue(batch_copy)) {
    // spin until write a value
  }
}
*/





} // end MonocularVO
#include "local_tracking_handler.hpp"

#include <memory>

namespace MonocularVO
{

LocalTrackingHandler::LocalTrackingHandler(
const MonocularVO::Params& params,
TypeCallbackTrack &callback_view_tracked)
: send_to_ros_interface{std::move(callback_view_tracked)},
  m_keep_tracking(true), m_params(params), m_frames(params),
  m_is_ref_frame_selected(false)
{
  // Local Handler Stuff:
  // * Local handler solves local BA problem
  // Batch :: [last_KF, N x KF,  curr_KF] --> Local Bundle Adjustment design
  //m_queue_batch_to_local_handler = std::make_shared<LockFreeQueueBatch>(30);
  //m_worker_local_handler = std::make_shared<LocalHandler>(params);
  //m_worker_local_handler->start(m_queue_batch_to_local_handler);
}

void LocalTrackingHandler::start(
  std::shared_ptr<LockFreeQueue> &queue_view_to_initialization)
{
  m_future_worker_local_tracking_handler = std::async(
      std::launch::async, &LocalTrackingHandler::track_frames,
       this,
       std::ref(queue_view_to_initialization));
}

LocalTrackingHandler::~LocalTrackingHandler()
{
  std::cout << "Shutdown LocalTrackingHandler~." << std::endl;
  m_future_worker_local_tracking_handler.get();
}

void LocalTrackingHandler::stop(){ m_keep_tracking = false;}


void
LocalTrackingHandler::track_frames(
  std::shared_ptr<LockFreeQueue> &queue_view_to_tracking)
{
  while (m_keep_tracking)
  {
    // Take curr_frame from queue
    FrameSharedPtr curr_frame(new Frame);
    while (!queue_view_to_tracking->try_dequeue(curr_frame)) {}

    m_frames.push_frame(curr_frame);

    // Only first iteration:
    if(!m_is_ref_frame_selected)
    {
      m_frames.set_curr_frame_is_ref_frame();
      m_is_ref_frame_selected = true;
      Vision::extract_features(curr_frame, m_params);
      continue;
    }

    // Tracking spins:
    if (m_is_ref_frame_selected)
    {
      track_p2d_optical_flow(25);
    }

    // Each new frame comes:
    m_frames.print_info();

    std::cout << "###########################"
                 "##########################" <<
                 "##########################" <<
                 "##########################" << std::endl;


  } // eof m_keep_tracking

}


void
LocalTrackingHandler::track_p2d_optical_flow(const int& window_size)
{
  FrameSharedPtr prev_frame = m_frames.get_prev_frame();
  FrameSharedPtr curr_frame = m_frames.get_curr_frame();

  // calculate optical flow
  std::vector<uchar> status;
  std::vector<float> err;
  cv::Size winSize = cv::Size(window_size,window_size);
  cv::TermCriteria termcrit=cv::TermCriteria(
      cv::TermCriteria::COUNT+cv::TermCriteria::EPS,
      50, 0.01);

  std::cout << "Optical flow processing on frames: " <<
    prev_frame->frame_id << "-" << curr_frame->frame_id << std::endl;

  auto start = std::chrono::steady_clock::now();
  cv::calcOpticalFlowPyrLK(prev_frame->image_gray,
                           curr_frame->image_gray,
                           prev_frame->keypoints_p2d,
                           curr_frame->keypoints_p2d,
                           status, err, winSize,
                           3, termcrit,
                           0, 0.001);
  auto end = std::chrono::steady_clock::now();
  /*std::cout << "Optical flow tooks: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
            << " millisecond." << std::endl;*/
  // Remove lost keypoints:
  int x = prev_frame->width;
  int y = prev_frame->height;
  int indexCorrection = 0;

  std::vector<cv::Point2f> deneme_ = prev_frame->keypoints_p2d;


  for( int i=0; i<status.size(); i++)
  {
    cv::Point2f pt = curr_frame->keypoints_p2d.at(
        i- indexCorrection);
    if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0)||pt.x>x||pt.y>y)
    {
      FrameSharedPtr ref_frame = m_frames.get_ref_frame();
      ref_frame->keypoints_p2d.erase(
          ref_frame->keypoints_p2d.begin() + i - indexCorrection);

      // Remove lost points in current frame
      curr_frame->keypoints_p2d.erase(
        curr_frame->keypoints_p2d.begin() + i - indexCorrection);

      indexCorrection++;
    }
  }

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
    frame->frame_id != map_->get_curr_frame()->frame_id)
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
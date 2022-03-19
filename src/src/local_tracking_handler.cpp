#include "local_tracking_handler.hpp"

#include <memory>

namespace MonocularVO
{

LocalTrackingHandler::LocalTrackingHandler(
const MonocularVO::Params& params,
TypeCallbackTrack &callback_view_tracked)
: send_to_ros_interface{std::move(callback_view_tracked)},
  m_keep_tracking(true), m_params(params), m_frames(params),
  m_is_ref_frame_selected(false), m_is_init_done(false)
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
  cv::namedWindow( "Local Feature Tracking", cv::WINDOW_FULLSCREEN);
  cv::moveWindow("Local Feature Tracking", 20,20);


  while (m_keep_tracking)
  {
    auto start_wait_for_frame = std::chrono::steady_clock::now();
    // Take curr_frame from queue
    FrameSharedPtr curr_frame(new Frame);
    while (!queue_view_to_tracking->try_dequeue(curr_frame)) {}
    m_frames.push_frame(curr_frame);
    auto end_wait_for_frame = std::chrono::steady_clock::now();
    std::cout << "Waiting for new frame tooks: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     end_wait_for_frame - start_wait_for_frame).count()
              << " millisecond." << std::endl;

    // Only first iteration:
    if(!m_is_ref_frame_selected)
    {
      m_frames.set_curr_frame_is_ref_frame();
      m_is_ref_frame_selected = true;
      m_tracked_p2d_ids.clear();
      Vision::extract_features(curr_frame, m_params);
      m_tracked_p2d_ids.resize(
        m_frames.get_ref_frame()->ref_frame_initial_observed_points_2d.size());
      std::iota (std::begin(m_tracked_p2d_ids),
            std::end(m_tracked_p2d_ids), 0);
      continue;
    }
    // Tracking spins:
    auto start_local_tracking_spin = std::chrono::steady_clock::now();
    if (m_is_ref_frame_selected)
    {
      track_observations_optical_flow(50, m_params.ransac_outlier_threshold);
      show_tracking(1.5);

      if (m_tracked_p2d_ids.size() > m_params.count_min_tracked)
      {
        // TODO:: if(is_init_done) {} else {}
        if (!m_is_init_done)
        {

        }

        if (m_is_init_done)
        {

        }

      } else if (m_tracked_p2d_ids.size() <=  m_params.count_min_tracked)
      {
        std::cout << "New local map detected." << std::endl;
        m_frames.set_curr_frame_is_ref_frame();
        m_is_ref_frame_selected = true;
        m_tracked_p2d_ids.clear();
        Vision::extract_features(curr_frame, m_params);
        m_tracked_p2d_ids.resize(
          m_frames.get_ref_frame()->ref_frame_initial_observed_points_2d.size());
        std::iota (std::begin(m_tracked_p2d_ids),
                  std::end(m_tracked_p2d_ids), 0);
        m_is_init_done = false;
      }
      auto end_local_tracking_spin = std::chrono::steady_clock::now();
      std::cout << "Tracking spin tooks: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(
                       end_local_tracking_spin - start_local_tracking_spin).count()
                << " millisecond." << std::endl;
    }

    // Each new frame comes:
    std::cout << "Reference frame id: " << m_frames.get_ref_frame()->frame_id << std::endl;
    std::cout << "Count curr tracked point count: " << m_tracked_p2d_ids.size() << std::endl;
    //m_frames.print_info();
    std::cout << "###########################"
                 "##########################" <<
                 "##########################" <<
                 "##########################" << std::endl;
  } // eof m_keep_tracking

  cv::destroyAllWindows();
}


void
LocalTrackingHandler::track_observations_optical_flow(const int& window_size,
                                                      const double& repr_threshold)
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

  cv::calcOpticalFlowPyrLK(prev_frame->image_gray,
                           curr_frame->image_gray,
                           prev_frame->keypoints_p2d,
                           curr_frame->keypoints_p2d,
                           status, err, winSize,
                           3, termcrit,
                           0, 0.001);
  // Remove lost keypoints:
  int x = prev_frame->width;
  int y = prev_frame->height;
  int indexCorrection = 0;
  for( int i=0; i<status.size(); i++)
  {
    cv::Point2f pt = curr_frame->keypoints_p2d.at(
        i- indexCorrection);
    if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0)||pt.x>x||pt.y>y)
    {
      // Remove lost points in ref frame keypoint ids
      m_tracked_p2d_ids.erase(
          m_tracked_p2d_ids.begin() + i - indexCorrection);
      // Remove lost points in prev frane in order to use in epipolar tracking refinement:
      prev_frame->keypoints_p2d.erase(
          prev_frame->keypoints_p2d.begin() + i - indexCorrection);
      // Remove lost points in current frame
      curr_frame->keypoints_p2d.erase(
        curr_frame->keypoints_p2d.begin() + i - indexCorrection);
      indexCorrection++;
    }
  }
  // Calculate Fundamental Matrix to refine tracked keypoints:
  cv::Mat inliers_F;
  cv::Mat F = cv::findFundamentalMat(
      prev_frame->keypoints_p2d,
      curr_frame->keypoints_p2d,
      cv::FM_RANSAC,
      repr_threshold,
      0.99, 500,
      inliers_F);
  // Calculate Essential matrix to refine tracked keypoints:
  cv::Mat inliers_E;
  cv::Mat E = cv::findEssentialMat(prev_frame->keypoints_p2d,
                                   curr_frame->keypoints_p2d,
                                   m_params.K,
                                   cv::RANSAC, 0.99,
                                   repr_threshold, inliers_E);

  // Remove bad points epipolar constraints:
  indexCorrection = 0;
  for(size_t i=0; i<inliers_E.rows; i++)
  {
    if (inliers_F.at<bool>(i,0) == false or
        inliers_E.at<bool>(i,0) == false)
    {
      // Remove lost points in ref frame keypoint ids
      m_tracked_p2d_ids.erase(
          m_tracked_p2d_ids.begin() + i - indexCorrection);
      // Remove lost points in prev frane
      prev_frame->keypoints_p2d.erase(
          prev_frame->keypoints_p2d.begin() + i - indexCorrection);
      // Remove lost points in current frame
      curr_frame->keypoints_p2d.erase(
          curr_frame->keypoints_p2d.begin() + i - indexCorrection);
      indexCorrection++;
    }
  }
}

void
LocalTrackingHandler::show_tracking(const float& downs_ratio)
{
  FrameSharedPtr ref_frame = m_frames.get_ref_frame();
  FrameSharedPtr curr_frame = m_frames.get_curr_frame();
  std::vector<cv::Point2f> ref_points;
  // Filter reference frame keypoints:
  for (int i=0; i < m_tracked_p2d_ids.size(); i++)
    ref_points.push_back(ref_frame->ref_frame_initial_observed_points_2d.at(m_tracked_p2d_ids.at(i)));
  // Current frame keypoints:
  std::vector<cv::Point2f> curr_points = m_frames.get_curr_frame()->keypoints_p2d;
  cv::Mat img_concat;
  cv::Mat img_show;

  cv::vconcat(ref_frame->image_gray_with_kpts,
              curr_frame->image_gray_with_kpts,
              img_concat);

  for (int i=0; i <curr_points.size(); i++)
  {
    cv::Point2f px_upper = ref_points[i];
    cv::Point2f px_lower = curr_points[i];
    px_lower.y += ref_frame->image_gray.rows;
    if (true)
    {
      cv::line(img_concat, px_upper, px_lower,
               cv::Scalar(0, 255, 0),
               1, cv::LINE_8);
    }

    cv::circle(img_concat, ref_points[i], 3,
               cv::Scalar(0, 0, 255),
               2, 4, 0);

    cv::Point2f p = curr_points[i];
    p.y += ref_frame->image_gray.rows;;
    cv::circle(img_concat, p, 4,
               cv::Scalar(0, 0, 255),
               3, 4, 0);
  }
  img_show = img_concat;
  cv::resize(img_show,
             img_show,
             cv::Size(m_frames.get_curr_frame()->width/downs_ratio,
                      m_frames.get_curr_frame()->height/downs_ratio),
             cv::INTER_LINEAR);
  cv::imshow("Local Feature Tracking",
             img_show);
  cv::waitKey(1);
}

bool
LocalTrackingHandler::is_tracking_ok()
{

  return false;
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
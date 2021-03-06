#include "local_tracking_handler.hpp"

#include <memory>

namespace MonocularVO
{

LocalTrackingHandler::LocalTrackingHandler(
const MonocularVO::Params& params,
TypeCallbackTrack &callback_view_tracked)
: send_to_ros_interface{std::move(callback_view_tracked)},
  m_keep_tracking(true), m_params(params), m_frames(params),
  m_is_ref_frame_selected(false), m_is_init_done(false),
  m_initializer(params), m_counter_p3d(0)
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
  cv::moveWindow("Local Feature Tracking", 0,0);

  bool print_tracking_info = false;
  cv::Mat t_all = cv::Mat::zeros(3,1,CV_64F);
  cv::Mat R_all = cv::Mat::eye(3,3,CV_64F);
  cv::Mat camera_pose = cv::Mat::eye(4, 4, CV_64F);

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


    // Only first iteration maybe requiring global initialization:
    if(!m_is_ref_frame_selected)
    {
      LocalTrackingHandler::make_reference_frame(curr_frame);
      m_is_init_done = false;
      if (print_tracking_info)
        print_tracking();
      continue;
    }
    // Tracking spins:
    auto start_local_tracking_spin = std::chrono::steady_clock::now();
    if (m_is_ref_frame_selected)
    {
      LocalTrackingHandler::track_observations_optical_flow(25,
                m_params.ransac_outlier_threshold);

      //LocalTrackingHandler::track_observations_descriptor_matching(
        //            m_params.ransac_outlier_threshold);

      m_tracking_evaluation = LocalTrackingHandler::eval_tracking(
          m_params.max_angular_px_disp,
                          10,
                          false);
      LocalTrackingHandler::show_tracking(1.5);

      //std::this_thread::sleep_for(3000000ms);

      m_tracking_evaluation.ready_for_trying_to_init = true;

      if (m_tracking_evaluation.is_tracking_ok)
      {
        cv::Mat R, t;
        Vision::pose_estimation_2d2d(m_frames.get_prev_frame()->keypoints_p2d,
                                     m_frames.get_curr_frame()->keypoints_p2d,
                                     m_params, R, t);

        // Scale can ben given from speed odometer or IMU | x = x0 + V*dt
        double scale = 1.0;
        cv::Mat T = cv::Mat::eye(4, 4, R.type());
        T(cv::Rect(0, 0, 3, 3)) = R * 1.0;
        T.col(3).rowRange(0, 3) = t * 1.0;
        camera_pose = camera_pose * T.inv();
        // print camera pose translation:
        std::cout << "Camera Pose Translation: " << std::endl;
        std::cout << camera_pose.col(3).rowRange(0, 3) << std::endl;



        // Tracking is ok | not initialized | ready try to init
        if (!m_is_init_done and m_tracking_evaluation.ready_for_trying_to_init)
        {
          /*
          std::vector<cv::Point2f> curr_tracked_p2d = m_frames.get_curr_frame()->keypoints_p2d;
          std::vector<cv::Point2f> ref_tracked_p2d;
          // Filter reference frame key-points:
          for (const int& id : m_vector_tracked_p3d_ids_local)
            ref_tracked_p2d.push_back(m_vector_ref_keypoints_p2d.at(id));
          // Try initialization:
          m_is_init_done = m_initializer.try_init(ref_tracked_p2d, curr_tracked_p2d,
                                                  m_vector_tracked_p3d_ids_global,
                                                  m_vector_p3d,
                                                  t);
                                                  */
          m_is_init_done = false;

          if (m_is_init_done)
          {
            using namespace std::chrono_literals;
            std::this_thread::sleep_for(10000ms);
            LocalTrackingHandler::show_tracking(1.2);
            std::cout << "Initialization done." << std::endl;
            std::this_thread::sleep_for(1000000ms);
          }


        }

        // Tracking is ok | initialized
        if (m_is_init_done)
        {
          std::vector<ObservationSharedPtr> new_obs = build_observations();
        }

      } // eof is_is_tracking_ok

      if (!m_tracking_evaluation.is_tracking_ok)
      {
        std::cout << "New local map detected." << std::endl;

        LocalTrackingHandler::make_reference_frame(curr_frame);
        m_is_init_done = false;
      }
      auto end_local_tracking_spin = std::chrono::steady_clock::now();
      std::cout << "Tracking spin tooks: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(
                       end_local_tracking_spin - start_local_tracking_spin).count()
                << " millisecond." << std::endl;
    }
    // Each new frame comes:
    std::cout << "Curr frame id: " << m_frames.get_curr_frame()->frame_id << std::endl;
    std::cout << "Reference frame id: " << m_frames.get_ref_frame()->frame_id << std::endl;
    // Build observations ##########################################################################
    if (print_tracking_info)
      print_tracking();


    std::cout << "Curr tracked landmark: " << m_vector_tracked_p3d_ids_global.size() << std::endl;

    //m_frames.print_info();
    std::cout << "\n###########################"
                 "##########################" <<
                 "##########################" <<
                 "##########################" << std::endl;
  } // eof m_keep_tracking

  cv::destroyAllWindows();
}


void
LocalTrackingHandler::track_observations_descriptor_matching(
  const double &repr_threshold)
{
  assert(prev_frame->keypoints.size() == prev_frame->keypoints_p2d.size());
  assert(prev_frame->keypoints.size() == m_vector_tracked_p3d_ids_local.size());
  assert(m_vector_tracked_p3d_ids_global.size() == m_vector_tracked_p3d_ids_local.size());

  FrameSharedPtr prev_frame = m_frames.get_prev_frame();
  FrameSharedPtr curr_frame = m_frames.get_curr_frame();

  Vision::extract_features(curr_frame, m_params);

  // print prev frame kpts:
//  std::cout << "Prev frame kpts: "  << std::endl;
//  Utils::print_keypoints_with_indexes(prev_frame->keypoints);
//  std::cout << "Curr frame kpts: "  << std::endl;
//  Utils::print_keypoints_with_indexes(curr_frame->keypoints);

  std::vector<cv::DMatch> matches;
  std::cout << "Match descriptors..." << std::endl;
  Vision::match_descriptors(prev_frame->keypoints,
                            curr_frame->keypoints,
                            prev_frame->descriptors,
                            curr_frame->descriptors,
                            matches,
                            m_params);

  assert(prev_frame->keypoints.size() == prev_frame->keypoints_p2d.size());
  assert(curr_frame->keypoints.size() == curr_frame->keypoints_p2d.size());

  std::cout << "Match count: " << matches.size() << std::endl;

  // Handle prev frame:
  std::vector<int> matched_ids_prev_frame;
  std::vector<int> not_matched_ids_prev_frame;
  for (const auto &match : matches)
    matched_ids_prev_frame.push_back(match.queryIdx);

  Utils::get_not_matched_kpt_ids(prev_frame, matches,
                                 not_matched_ids_prev_frame,
                                 true);

//  std::cout << "Count not matched prev frame keypoints: " <<
//      not_matched_ids_prev_frame.size() << std::endl;

//  std::cout << "Matched ids of prev frame: ";
//  Utils::print_vector_elements(matched_ids_prev_frame);
//
//  std::cout << "Not matched ids of prev frame: " << std::endl;
//  Utils::print_vector_elements(not_matched_ids_prev_frame);

  // New prev frame keypoints:
  std::vector<cv::KeyPoint> new_prev_frame_keypoints;
  std::vector<cv::Point2f> new_prev_frame_keypoints_p2d;
  for (const int& id : matched_ids_prev_frame)
  {
    new_prev_frame_keypoints.push_back(prev_frame->keypoints[id]);
    new_prev_frame_keypoints_p2d.push_back(prev_frame->keypoints_p2d[id]);
  }
  prev_frame->keypoints = new_prev_frame_keypoints;
  prev_frame->keypoints_p2d = new_prev_frame_keypoints_p2d;


  // Handle curr frame:
  std::vector<int> matched_ids_curr_frame;
  for (const auto &match : matches)
    matched_ids_curr_frame.push_back(match.trainIdx);

//  std::cout << "Matched ids of curr frame: ";
//  Utils::print_vector_elements(matched_ids_curr_frame);

  // New curr frame keypoints:
  std::vector<cv::KeyPoint> new_curr_frame_keypoints;
  std::vector<cv::Point2f> new_curr_frame_keypoints_p2d;
  for (const int& id : matched_ids_curr_frame)
  {
    new_curr_frame_keypoints.push_back(curr_frame->keypoints[id]);
    new_curr_frame_keypoints_p2d.push_back(curr_frame->keypoints_p2d[id]);
  }
  curr_frame->keypoints = new_curr_frame_keypoints;
  curr_frame->keypoints_p2d = new_curr_frame_keypoints_p2d;

//  std::cout << "Prev frame kpts: ";
//  Utils::print_keypoints_with_indexes(prev_frame->keypoints);
//  std::cout << "Curr frame kpts: ";
//  Utils::print_keypoints_with_indexes(curr_frame->keypoints);

  // clear lost information:
  Utils::remove_lost_landmark_ids(not_matched_ids_prev_frame,
                                  m_vector_tracked_p3d_ids_local);
  Utils::remove_lost_landmark_ids(not_matched_ids_prev_frame,
                                  m_vector_tracked_p3d_ids_global);



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
  int indexCorrection = 0;
  for(size_t i=0; i<inliers_E.rows; i++)
  {
    if (inliers_F.at<bool>(i,0) == false or
        inliers_E.at<bool>(i,0) == false)
    {
      // Remove lost points in ref frame keypoint ids
      m_vector_tracked_p3d_ids_global.erase(
          m_vector_tracked_p3d_ids_global.begin() + i - indexCorrection);
      m_vector_tracked_p3d_ids_local.erase(
          m_vector_tracked_p3d_ids_local.begin() + i - indexCorrection);

      curr_frame->keypoints_p2d.erase(
          curr_frame->keypoints_p2d.begin() + i - indexCorrection);

      curr_frame->keypoints.erase(
          curr_frame->keypoints.begin() + i - indexCorrection);

      indexCorrection++;
    }
  }

  std::cout << "Count curr frame keypoints: " << curr_frame->keypoints_p2d.size() << "\n" << std::endl;

//  std::cout << "\nm_vector_tracked_p3d_ids_local: ";
//  Utils::print_vector_elements(m_vector_tracked_p3d_ids_local);
//  std::cout << "m_vector_tracked_p3d_ids_global: ";
//  Utils::print_vector_elements(m_vector_tracked_p3d_ids_global);


  cv::Mat new_descriptor;
  Vision::desc_keypoints(m_params, curr_frame->keypoints, new_descriptor,
                         curr_frame->image_gray);
  curr_frame->descriptors.release();
  curr_frame->descriptors = new_descriptor.clone();


}


void
LocalTrackingHandler::make_reference_frame(FrameSharedPtr& curr_frame)
{
  m_frames.set_curr_frame_is_ref_frame();
  m_is_ref_frame_selected = true;
  m_vector_tracked_p3d_ids_local.clear();
  m_vector_tracked_p3d_ids_global.clear();
  m_vector_ref_keypoints.clear();
  m_vector_ref_keypoints_p2d.clear();

  curr_frame->descriptors.release();
  curr_frame->descriptors = cv::Mat();

  Vision::extract_features(curr_frame, m_params);

  m_vector_tracked_p3d_ids_local.resize(
      m_frames.get_ref_frame()->keypoints.size());
  std::iota (std::begin(m_vector_tracked_p3d_ids_local),
            std::end(m_vector_tracked_p3d_ids_local), 0);

  m_vector_tracked_p3d_ids_global.resize(
      m_frames.get_ref_frame()->keypoints.size());
  std::iota (std::begin(m_vector_tracked_p3d_ids_global),
            std::end(m_vector_tracked_p3d_ids_global), m_counter_p3d);

  // At this point Point3Ds can be provided from LiDAR range sensor
  // in order to absorb scale stuff xD
  for (int i=0; i<m_frames.get_ref_frame()->keypoints.size(); i++)
  {
    m_vector_p3d.emplace_back(static_cast<float>(m_counter_p3d),
                                       static_cast<float>(m_counter_p3d),
                                       static_cast<float>(m_counter_p3d));
    m_vector_ref_keypoints.push_back(curr_frame->keypoints[i]);
    m_vector_ref_keypoints_p2d.push_back(curr_frame->keypoints[i].pt);
    m_counter_p3d += 1;
  }
}


std::vector<ObservationSharedPtr>
LocalTrackingHandler::build_observations()
{
  std::vector<ObservationSharedPtr> new_obs;
  int id_frame = m_frames.get_curr_frame()->frame_id;
  for (int i=0; i< m_vector_tracked_p3d_ids_global.size(); i++)
  {
    int id_p3d = m_vector_tracked_p3d_ids_global[i];
    cv::Point2d p2d = m_frames.get_curr_frame()->keypoints_p2d.at(i);
    ObservationSharedPtr observation = std::make_shared<Observation>(
        id_frame, id_p3d, cv::Vec6d {0,0,0,0,0,0},
        cv::Mat(), p2d,
        m_vector_p3d[id_p3d],
        m_frames.get_curr_frame()->is_keyframe,
        m_frames.get_curr_frame()->is_ref_frame,
        false, m_is_init_done);
    new_obs.push_back(observation);
  }
  return new_obs;
}


LocalTrackingHandler::TrackingEvaluation
LocalTrackingHandler::eval_tracking(const double& avg_px_dis_threshold,
                                    const int& count_diff_frame_threshold,
                                    const bool& print_info)
{
  // Define initial states:
  TrackingEvaluation tracking_evaluation;
  tracking_evaluation.is_tracking_ok = true;
  tracking_evaluation.ready_for_trying_to_init = false;
  tracking_evaluation.average_ang_px_displacement = 0;

  // Evaluation of states:
  if(m_vector_tracked_p3d_ids_local.size() <= m_params.count_min_tracked)
    tracking_evaluation.is_tracking_ok = false;

  if(tracking_evaluation.is_tracking_ok)
  {
    FrameSharedPtr ref_frame = m_frames.get_ref_frame();
    FrameSharedPtr curr_frame = m_frames.get_curr_frame();
    std::vector<cv::Point2f> ref_points;
    // Filter reference frame keypoints:
    for (int i=0; i < m_vector_tracked_p3d_ids_local.size(); i++)
      ref_points.push_back(m_vector_ref_keypoints_p2d.at(m_vector_tracked_p3d_ids_local.at(i)));

    tracking_evaluation.average_ang_px_displacement = Vision::average_ang_px_displacement(
        ref_points,
        curr_frame->keypoints_p2d,
        ref_frame->height,
        ref_frame->width);

    // Calculate frame diff:
    int diff_frame_count = curr_frame->frame_id - ref_frame->frame_id;

    // Calculate average angular px displacement:
    if (tracking_evaluation.average_ang_px_displacement > avg_px_dis_threshold
        and diff_frame_count > count_diff_frame_threshold
        and
        m_vector_tracked_p3d_ids_local.size() > 50)
      tracking_evaluation.ready_for_trying_to_init = true;
    else
      tracking_evaluation.ready_for_trying_to_init = false;
  }

  if (print_info)
  {
    std::cout << "Tracking evaluation report:" << std::endl;
    std::cout << "***************************************************" << std::endl;
    std::cout << "Count curr tracked point count: " << m_vector_tracked_p3d_ids_local.size() << std::endl;
    std::cout << "Ready for trying initialization: " <<
        tracking_evaluation.ready_for_trying_to_init << std::endl;
    std::cout << "Average ang px displacement: " <<
        tracking_evaluation.average_ang_px_displacement << std::endl;
    std::cout << "***************************************************" << std::endl;
  }

  return tracking_evaluation;
}


void
LocalTrackingHandler::track_observations_optical_flow(const int& window_size,
                                                      const double& repr_threshold)
{
  FrameSharedPtr prev_frame = m_frames.get_prev_frame();
  FrameSharedPtr curr_frame = m_frames.get_curr_frame();

  assert(prev_frame->keypoints.size() == prev_frame->keypoints_p2d.size());
  assert(prev_frame->keypoints.size() == m_vector_tracked_p3d_ids_local.size());
  assert(m_vector_tracked_p3d_ids_global.size() == m_vector_tracked_p3d_ids_local.size());

  // calculate optical flow
  std::vector<uchar> status;
  std::vector<float> err;
  cv::Size winSize = cv::Size(window_size,window_size);
  cv::TermCriteria termcrit=cv::TermCriteria(
      cv::TermCriteria::COUNT+cv::TermCriteria::EPS,
      30, 0.01);

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
      m_vector_tracked_p3d_ids_global.erase(
          m_vector_tracked_p3d_ids_global.begin() + i - indexCorrection);
      m_vector_tracked_p3d_ids_local.erase(
          m_vector_tracked_p3d_ids_local.begin() + i - indexCorrection);
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

  // Remove bad points epi-polar constraints:
  indexCorrection = 0;
  for(size_t i=0; i<inliers_E.rows; i++)
  {
    if (inliers_F.at<bool>(i,0) == false or
        inliers_E.at<bool>(i,0) == false)
    {
      // Remove lost points in ref frame keypoint ids
      m_vector_tracked_p3d_ids_global.erase(
          m_vector_tracked_p3d_ids_global.begin() + i - indexCorrection);
      m_vector_tracked_p3d_ids_local.erase(
          m_vector_tracked_p3d_ids_local.begin() + i - indexCorrection);
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
  cv::Mat img_show_ref;
  cv::Mat img_show_curr;
  ref_frame->img_colored.copyTo(img_show_ref);
  curr_frame->img_colored.copyTo(img_show_curr);

  std::string count_curr_track = std::to_string(m_vector_tracked_p3d_ids_local.size());

  cv::putText(img_show_ref,
              count_curr_track,
              cv::Point(curr_frame->width/1.5, curr_frame->height/5),
              cv::FONT_HERSHEY_DUPLEX,
              4, CV_RGB(255, 0, 0), 3);

  if (m_tracking_evaluation.ready_for_trying_to_init)
  {
    cv::putText(img_show_ref,
                "Ready to try init: True",
                cv::Point(curr_frame->width/1.5, curr_frame->height/1.1),
                cv::FONT_HERSHEY_DUPLEX,
                1.5, CV_RGB(255, 0, 0), 2);

    cv::putText(img_show_curr,
                "Ready to try init: True",
                cv::Point(curr_frame->width/1.5, curr_frame->height/1.1),
                cv::FONT_HERSHEY_DUPLEX,
                1.5, CV_RGB(255, 0, 0), 2);
  }
  if (!m_tracking_evaluation.ready_for_trying_to_init)
  {
    cv::putText(img_show_ref,
                "Ready to try init: False",
                cv::Point(curr_frame->width/1.5, curr_frame->height/1.1),
                cv::FONT_HERSHEY_DUPLEX,
                1.5, CV_RGB(255, 0, 0), 2);

    cv::putText(img_show_curr,
                "Ready to try init: False",
                cv::Point(curr_frame->width/1.5, curr_frame->height/1.1),
                cv::FONT_HERSHEY_DUPLEX,
                1.5, CV_RGB(255, 0, 0), 2);
  }

  std::vector<cv::Point2f> ref_points;
  // Filter reference frame keypoints:
  for (int i=0; i < m_vector_tracked_p3d_ids_local.size(); i++)
    ref_points.push_back(m_vector_ref_keypoints_p2d.at(m_vector_tracked_p3d_ids_local.at(i)));
  // Current frame keypoints:
  std::vector<cv::Point2f> curr_points = m_frames.get_curr_frame()->keypoints_p2d;
  cv::Mat img_concat;
  cv::Mat img_show;

  cv::vconcat(img_show_ref,
              img_show_curr,
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
               1, 4, 0);

    cv::Point2f p = curr_points[i];
    p.y += ref_frame->image_gray.rows;;
    cv::circle(img_concat, p, 4,
               cv::Scalar(0, 0, 255),
               1, 4, 0);
  }
  img_show = img_concat;
  cv::resize(img_show,
             img_show,
             cv::Size(m_frames.get_curr_frame()->width/downs_ratio,
                      m_frames.get_curr_frame()->height/downs_ratio),
             cv::INTER_LINEAR);
  cv::imshow("Local Feature Tracking",
             img_show);
  cv::waitKey(1000);
}


void
LocalTrackingHandler::print_tracking()
{
  if (m_vector_tracked_p3d_ids_local.size() ==
      m_vector_tracked_p3d_ids_global.size())
  {
    std::cout << "m_vector_tracked_p3d_ids_local:" << std::endl;
    for (int i=0; i< m_vector_tracked_p3d_ids_local.size(); i++)
    {
      std::cout << "ID Point3D: " << m_vector_tracked_p3d_ids_global[i] <<
        " Point2D: " << m_frames.get_curr_frame()->keypoints_p2d[i] << " | ";
    } std::cout << " \n"<<  std::endl;
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
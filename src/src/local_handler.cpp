#include "local_handler.hpp"

using namespace std::chrono;

namespace BundleAdjustment
{


BundleAdjustment::LocalHandler::LocalHandler(
  const BundleAdjustment::Params &params)
: params_(params),
  keep_local_handling_(true)
{}


BundleAdjustment::LocalHandler::~LocalHandler()
{
  std::cout << "Shutdown ~LocalHandler." << std::endl;
  future_worker_local_handler_.get();
}

void BundleAdjustment::LocalHandler::start(
  std::shared_ptr<LockFreeQueueBatch> &queue_batch_to_local_handler)
{
  future_worker_local_handler_ = std::async(std::launch::async,
                      &LocalHandler::handle,
                      this,
                      std::ref(queue_batch_to_local_handler));
}

void
BundleAdjustment::LocalHandler::handle(
  std::shared_ptr<LockFreeQueueBatch> &queue_batch_to_local_handler)
{
  Batch batch;
  std::mutex mutex;

  FrameSharedPtr based_frame(new Frame);

  while (keep_local_handling_)
  {
    // Wait for new batch
    while (!queue_batch_to_local_handler->try_dequeue(
        batch)) {}

    mutex.lock();
    //std::cout << "Batch is received to local handler." << std::endl;
    auto start = high_resolution_clock::now();
    LocalObservations local_observations = build_local_observations(batch);
    BundleAdjustment::Optimization::LocalObservations local_observations_opt = Optimization::solve_local_ba(local_observations, params_.K);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    std::cout << "Time taken by function: "
         << duration.count() << " milliseconds" << std::endl;


    double reprojection_err = 0;
    for (int i=0; i<local_observations_opt.camera_poses.size(); i++)
    {
      // Transformation world to camera frame:
      cv::Vec3d rvec(local_observations_opt.camera_poses[i][0], local_observations_opt.camera_poses[i][1], local_observations_opt.camera_poses[i][2]);
      cv::Mat R = cv::Mat(3, 3, 6);
      cv::Rodrigues(rvec, R);
      cv::Mat t(3, 1, 6);
      t = (local_observations_opt.camera_poses[i][3], local_observations_opt.camera_poses[i][4], local_observations_opt.camera_poses[i][5]);
      for (int j=0; j<batch[i]->keypoints_pt.size(); j++)
      {
        cv::Point2d projected_point = Utils::project_point(R, t, params_.K, local_observations_opt.points3D[j]);
        cv::Point2d real_point = batch[i]->keypoints_pt[j];
        reprojection_err += sqrt(pow(real_point.x - projected_point.x, 2) + pow(real_point.y - projected_point.y, 2));
        //std::cout << "Camera id: " << i << " | projected point: " << projected_point << " | real_point:" << real_point << std::endl;
      }
    }

    reprojection_err /= static_cast<double>(local_observations.camera_poses.size()*local_observations.points3D.size());
    std::cout << "Re-projection error after local optimization: " << reprojection_err << " px." << std::endl;

    std::cout << "##########################################"
                 "##############################################" << std::endl;
    mutex.unlock();


/*    mutex.lock();
    map_local_->push_key_frames(batch);
    map_local_->print_curr_batch_info();

    FrameSharedPtr prev_key_frame = map_local_->get_prev_key_frame();
    FrameSharedPtr curr_key_frame = map_local_->get_curr_key_frame();

    MatchKeyFrameSharedPtr match = Vision::match_key_frames(params_, prev_key_frame, curr_key_frame);

    std::cout << "Initialization is done." << std::endl;
    std::cout << "####################################"
                 "#########################################" << std::endl;

    mutex.unlock();*/


/*    //TODO:: Local Edge oluştur

    auto start = high_resolution_clock::now();
    map_local_->solve(batch);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    std::cout << "Time taken by function: " <<
              duration.count() << " milliseconds." << std::endl;
    //TODO:: Build graph, then solve local BA!*/
  }
}



void
BundleAdjustment::LocalHandler::stop()
{
  keep_local_handling_=false;
}


BundleAdjustment::LocalObservations
BundleAdjustment::LocalHandler::build_local_observations(Batch& batch)
{

  float ransac_threshold = 0.45;
 // Refine epipolar again with lower threshold
 cv::Mat E_init, F_init;
  for (int i=0; i<batch.size()-1; i++)
  {
    FrameSharedPtr frame_first = batch[i];
    FrameSharedPtr frame_second = batch[i+1];

    // Calculate fundamental matrix
    cv::Mat inliers_F;
    cv::Mat F = Vision::get_F(frame_first, frame_second,
                        ransac_threshold, inliers_F);
    // Calculate essential matrix
    cv::Mat inliers_E;
    cv::Mat E = Vision::get_E(frame_first, frame_second,
            ransac_threshold, inliers_E, params_.K);

    if (i == 0)
    {
      E_init = E;
      F_init = F;
    }

    int indexCorrection = 0;
    for(size_t i=0; i<inliers_E.rows; i++)
    {
      if (inliers_F.at<bool>(i,0) == false or
          inliers_E.at<bool>(i,0) == false)
      {
        for (auto it=batch.begin();
             it!=batch.end(); it++)
        {
          FrameSharedPtr view = *it;
          view->keypoints_pt.erase(view->keypoints_pt.begin() + i - indexCorrection);
        }
        indexCorrection++;
      }
    }
  }
  std::cout << "After batch epipolar "
   "refinement kpt count:" << batch.back()->keypoints_pt.size()
   << "\n" << std::endl;

  // Assumption
  // - All cameras have the same and known camera matrix.
  // - All points are visible on all camera views.

  // Calculate poses and 3D points incremental:
  cv::Mat R, t, Rt;
  FrameSharedPtr frame_first = batch[0];
  FrameSharedPtr frame_second = batch[1];
  Vision::recover_pose(frame_first, frame_second,
                       F_init, E_init,
                       R, t, params_.K);

  cv::Mat rvec;
  cv::Rodrigues(R, rvec);
  std::vector<cv::Vec6d> cameras(batch.size());
  cameras[0] = cv::Vec6d {0,0,0,0,0,0}; // World frame is the camera origin
  cameras[1] = cv::Vec6d {rvec.at<double>(0), rvec.at<double>(1),
                          rvec.at<double>(2),
                          t.at<double>(0,0), t.at<double>(1,0),
                          t.at<double>(2,0)};
  cv::hconcat(R, t, Rt);
  cv::Mat P0 = params_.K * cv::Mat::eye(3, 4, 6);
  cv::Mat P1 = params_.K * Rt, mat_points3D;
  cv::triangulatePoints(P0, P1,
                        frame_first->keypoints_pt,
                        frame_second->keypoints_pt,
                        mat_points3D);
  std::vector<cv::Point3d> vector_points3D(mat_points3D.cols);
  mat_points3D.row(0) = mat_points3D.row(0) / mat_points3D.row(3);
  mat_points3D.row(1) = mat_points3D.row(1) / mat_points3D.row(3);
  mat_points3D.row(2) = mat_points3D.row(2) / mat_points3D.row(3);
  mat_points3D.row(3) = 1;
  for (int c = 0; c < mat_points3D.cols; c++)
    vector_points3D[c] = cv::Point3d(mat_points3D.col(c).rowRange(0, 3));

  if (batch.size() > 2)
  {
    for (int i = 2; i < batch.size(); i++)
    {
      cv::Mat rvec_, t_;
      cv::solvePnP(vector_points3D, batch[i]->keypoints_pt,
                   params_.K, cv::noArray(), rvec_, t_);
      cameras[i] = cv::Vec6d {rvec_.at<double>(0), rvec_.at<double>(1),
                              rvec_.at<double>(2), t_.at<double>(0,0),
                              t_.at<double>(1,0), t_.at<double>(2,0)};
    }
  }

  // Check initialization results:
  double reprojection_err = 0;
  for (int i=0; i<cameras.size(); i++)
  {
    // Transformation world to camera frame:
    cv::Vec3d rvec(cameras[i][0], cameras[i][1], cameras[i][2]);
    cv::Mat R = cv::Mat(3, 3, 6);
    cv::Rodrigues(rvec, R);
    cv::Mat t(3, 1, 6);
    t = (cameras[i][3], cameras[i][4], cameras[i][5]);
    for (int j=0; j<batch[i]->keypoints_pt.size(); j++)
    {
      cv::Point2d projected_point = Utils::project_point(R, t, params_.K, vector_points3D[j]);
      cv::Point2d real_point = batch[i]->keypoints_pt[j];
      reprojection_err += sqrt(pow(real_point.x - projected_point.x, 2) + pow(real_point.y - projected_point.y, 2));
      //std::cout << "Camera id: " << i << " | projected point: " << projected_point << " | real_point:" << real_point << std::endl;
    }
  }

  reprojection_err /= static_cast<double>(cameras.size()*vector_points3D.size());
  std::cout << "Re-projection error before local optimization: " << reprojection_err << " px." << std::endl;

  // Load 2D points observed from multiple views:
  std::vector<std::vector<cv::Point2d>> xs;
  for (FrameSharedPtr &frame : batch) {
    std::vector<cv::Point2d> pts;
    for (const cv::Point2f &pt :frame->keypoints_pt) {
      pts.push_back(pt);
    }
    xs.push_back(pts);
  }
  assert(vector_points3D.size() == xs[0].size());

  LocalObservations local_observations;
  local_observations.points3D = vector_points3D;
  local_observations.camera_poses = cameras;
  local_observations.points2D = xs;

  return local_observations;
}


}
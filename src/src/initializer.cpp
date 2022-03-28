//
// Created by goktug on 20.03.2022.
//

#include "initializer.hpp"

namespace MonocularVO
{
Initializer::Initializer(const MonocularVO::Params& params)
: m_params(params)
{}



bool
Initializer::try_init(FrameSharedPtr& ref_frame,
                      FrameSharedPtr& curr_frame,
                      std::vector<int>&vector_tracked_p3d_ids_local,
                      std::vector<int>&vector_tracked_p3d_ids_global,
                      std::vector<cv::Point3d>& vector_initial_p3d,
                      const double& scale)
{
  if (curr_frame->is_ref_frame)
    return false;

  std::cout << "Doing initialization." << std::endl;

  std::vector<cv::Point2f> tracked_ref_keypoints;
  for (const int& tracked_id : vector_tracked_p3d_ids_local)
    tracked_ref_keypoints.push_back(ref_frame->keypoints[tracked_id].pt);
  std::vector<cv::Point2f> tracked_curr_keypoints = curr_frame->keypoints_p2d;

  if (tracked_ref_keypoints.size() != tracked_curr_keypoints.size())
    std::cout << "Error." << std::endl;

  // Firstly, undistort points:
  cv::undistortPoints(tracked_ref_keypoints,
                      tracked_ref_keypoints,
                      m_params.K,
                      m_params.mat_dist_coeff);
  cv::undistortPoints(tracked_curr_keypoints,
                      tracked_curr_keypoints,
                      m_params.K,
                      m_params.mat_dist_coeff);
  // Calculate relative pose
  cv::Mat E, inlier_mask;
  bool use_5pt = false;
  if (use_5pt)
  {
    E = cv::findEssentialMat(tracked_ref_keypoints,
                             tracked_curr_keypoints, m_params.K,
                             cv::RANSAC, 0.99, 1, inlier_mask);
  }
  else
  {
    cv::Mat F = cv::findFundamentalMat(tracked_ref_keypoints,
                                       tracked_curr_keypoints,
                                       cv::FM_RANSAC,
                                       0.01,0.99,
                                       inlier_mask);
    E = m_params.K.t() * F * m_params.K;
  }

  double f = m_params.K.at<double>(1,1);
  double cx = m_params.K.at<double>(0,2);
  double cy = m_params.K.at<double>(1,2);
  cv::Point2d c(cx, cy);
  // Parent frame is reference frame; child is current frame:
  cv::Mat R, t;
  int inlier_num = cv::recoverPose(
      E,tracked_ref_keypoints,tracked_curr_keypoints,
      R, t,f,c, inlier_mask);

  // Scale can ben given from speed odometer or IMU | x = x0 + V*dt
  // R_curr = R*R_curr;  t_curr = t_curr + scale*(R_curr*t);

  if (inlier_num > 300)
  {
    cv::Mat T = cv::Mat::eye(4, 4, R.type());
    T(cv::Rect(0, 0, 3, 3)) = R * 1.0;
    T.col(3).rowRange(0, 3) = t * 1.0;

    // Reconstruct 3D points (triangulation)
    cv::Mat P0 = m_params.K * cv::Mat::eye(3, 4, CV_64F);
    cv::Mat Rt, X;
    cv::hconcat(R, t, Rt);
    cv::Mat P1 = m_params.K * Rt;
    cv::triangulatePoints(P0, P1, tracked_ref_keypoints,
                          tracked_curr_keypoints, X);
    // Homogeous to cartesian:
    X.row(0) = X.row(0) / X.row(3);
    X.row(1) = X.row(1) / X.row(3);
    X.row(2) = X.row(2) / X.row(3);
    X.row(3) = 1;

    std::vector<cv::Point3d> vec_p3d;
    for (int i=0; i<X.rows; i++)
    {
      cv::Point3d p3d(X.at<double>(i, 0),
                      X.at<double>(i, 1),
                      X.at<double>(i, 2));
      vec_p3d.push_back(p3d);
    }

    int indexCorrection = 0;
    for( int i=0; i<X.rows; i++)
    {
      if (inlier_mask.at<uchar>(i) == 0)
      {
        vec_p3d.erase(
            vec_p3d.begin() + i - indexCorrection);
        // Remove lost points in ref frame keypoint ids
        vector_tracked_p3d_ids_global.erase(
            vector_tracked_p3d_ids_global.begin() + i - indexCorrection);
        vector_tracked_p3d_ids_local.erase(
            vector_tracked_p3d_ids_local.begin() + i - indexCorrection);
        // Remove lost points in current frame
        curr_frame->keypoints_p2d.erase(
            curr_frame->keypoints_p2d.begin() + i - indexCorrection);
        tracked_curr_keypoints.erase(
            tracked_curr_keypoints.begin() + i - indexCorrection);
        indexCorrection++;
      }
    }

    // Initialization result:
    for (int i=0; i<vector_tracked_p3d_ids_global.size(); i++)
    {
      int id = vector_tracked_p3d_ids_global[i];
      // Add to map:
      vector_initial_p3d[id] = vec_p3d[i];
    }

    // Evaluate the reprojection error:
    double reproj_error = 0;
    std::vector<cv::Point2d> reproj_points;
    for (int i=0; i<vec_p3d.size(); i++)
    {
      cv::Point3d p3d = vec_p3d[i];
      cv::Point2d p2d = Utils::project_point(R,t,m_params.K,p3d);
      reproj_points.push_back(p2d);
      reproj_error += sqrt(pow(p2d.x - tracked_curr_keypoints[i].x, 2) +
                           pow(p2d.y - tracked_curr_keypoints[i].y, 2));
    }

    std::cout << "Reproj error: " << reproj_error << std::endl;
    std::cout << "Count inliers: " << inlier_num << std::endl;
    std::cout << "Tracked Point3D count after initialization: " << vec_p3d.size() << std::endl;
    std::cout << T.inv() << std::endl;
    using namespace std::chrono_literals;
   std::this_thread::sleep_for(100000ms);
  }




  return false;
}

}  // namespace MonocularVO

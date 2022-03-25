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
Initializer::try_init(FrameSharedPtr &ref_frame,
                      FrameSharedPtr &curr_frame,
                      std::vector<int> &tracked_p2d_ids,
                      const double& scale)
{
  if (curr_frame->is_ref_frame)
    return false;

  std::cout << "Doing initialization." << std::endl;

  std::vector<cv::Point2f> tracked_ref_keypoints;
  for (const int& tracked_id : tracked_p2d_ids)
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
                                       1, 0.99,
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

  using namespace std::chrono_literals;
  std::this_thread::sleep_for(100000ms);

  // Scale can ben given from speed odometer or IMU | x = x0 + V*dt
  // R_curr = R*R_curr;  t_curr = t_curr + scale*(R_curr*t);

/*  if (inlier_num > 10)
  {
    cv::Mat T = cv::Mat::eye(4, 4, R.type());
    T(cv::Rect(0, 0, 3, 3)) = R * 1.0;
    T.col(3).rowRange(0, 3) = t * 1.0;
  }*/

  return false;
}

}

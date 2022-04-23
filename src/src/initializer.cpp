//
// Created by goktug on 20.03.2022.
//

#include "initializer.hpp"

namespace MonocularVO
{
Initializer::Initializer(const MonocularVO::Params& params)
: m_params(params)
{
}



bool
Initializer::try_init(std::vector<cv::Point2f>& kpts_ref,
                      std::vector<cv::Point2f>& kpts_curr,
                      std::vector<int>&vector_tracked_p3d_ids_global,
                      std::vector<cv::Point3f>& vector_p3d_global,
                      cv::Mat& t_all)
{
  std::cout << "Doing initialization." << std::endl;
  double fx = m_params.K.at<double>(0,0);
  double fy = m_params.K.at<double>(1,1);
  double cx = m_params.K.at<double>(0,2);
  double cy = m_params.K.at<double>(1,2);
  double f = m_params.K.at<double>(1,1);
  cv::Point2d c(cx, cy);
  // Undistorted points:
  // cv::undistortPoints(kpts_ref, kpts_ref_undistorted_norm_inlier,
  // m_params.K, m_params.mat_dist_coeff, cv::noArray(), m_params.K);
  std::vector<cv::Point2f> kpts_ref_undistorted_norm;
  std::vector<cv::Point2f> kpts_curr_undistorted_norm;
  cv::undistortPoints(kpts_ref, kpts_ref_undistorted_norm, m_params.K,
                      m_params.mat_dist_coeff);
  cv::undistortPoints(kpts_curr, kpts_curr_undistorted_norm, m_params.K,
                      m_params.mat_dist_coeff);
  std::vector<cv::Point2f> kpts_ref_undistorted_focal;
  std::vector<cv::Point2f> kpts_curr_undistorted_focal;
  auto norm_to_focal = [&](const cv::Point2f& p) {
    return cv::Point2f(p.x * fx + cx, p.y * fy + cy);
  };
  std::transform(kpts_ref_undistorted_norm.begin(), kpts_ref_undistorted_norm.end(),
                 std::back_inserter(kpts_ref_undistorted_focal), norm_to_focal);
  std::transform(kpts_curr_undistorted_norm.begin(), kpts_curr_undistorted_norm.end(),
                 std::back_inserter(kpts_curr_undistorted_focal), norm_to_focal);

  cv::Mat inlier_mask;
  cv::Mat F = cv::findFundamentalMat(kpts_ref_undistorted_focal,
                                     kpts_curr_undistorted_focal,
                                     cv::FM_RANSAC,
                                     1,0.99,
                                     inlier_mask);
  cv::Mat E = m_params.K.t() * F * m_params.K;
  // Essential matrix cannot zero matrix:
  std::cout << "E = " << std::endl << E << std::endl;
  cv::Mat R, t;

  int count_inlier = cv::recoverPose(
      E, kpts_ref_undistorted_norm,
      kpts_curr_undistorted_norm, R, t,
      f, c);


  return false;
}

}  // namespace MonocularVO

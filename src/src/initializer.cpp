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
Initializer::try_init(std::vector<cv::Point2f>& kpts_ref,
                      std::vector<cv::Point2f>& kpts_curr,
                      std::vector<int>&vector_tracked_p3d_ids_global,
                      std::vector<cv::Point3d>& vector_p3d_global)
{
  std::cout << "Doing initialization." << std::endl;
  double fx = m_params.K.at<double>(0,0);
  double fy = m_params.K.at<double>(1,1);
  double cx = m_params.K.at<double>(0,2);
  double cy = m_params.K.at<double>(1,2);
  // Undistort points:
  std::vector<cv::Point2f> kpts_ref_undistorted;
  std::vector<cv::Point2f> kpts_curr_undistorted;
  cv::undistortPoints(kpts_ref, kpts_ref_undistorted, m_params.K,
                      m_params.mat_dist_coeff, cv::Mat());

  cv::undistortPoints(kpts_curr, kpts_curr_undistorted, m_params.K,
                      m_params.mat_dist_coeff, cv::Mat());

  cv::Mat E, R, t, mask;

  // finde Essential Matrix
  E = cv::findEssentialMat(kpts_curr_undistorted, kpts_ref_undistorted,
                           m_params.K, cv::RANSAC, 0.999, 1.0, mask);

  double f = m_params.K.at<double>(1,1);
  cv::Point2d c(cx, cy);
  // Parent frame is reference frame; child is current frame:
  int inlier_num = cv::recoverPose(
      E,kpts_ref,kpts_curr,
      R, t,f,c, mask);

  // print inlier number:
  std::cout << "Inlier number: " << inlier_num << std::endl;

  // Scale can ben given from speed odometer or IMU | x = x0 + V*dt
  // R_curr = R*R_curr;  t_curr = t_curr + scale*(R_curr*t);

  if (inlier_num > 50) {

    cv::Mat T = cv::Mat::eye(4, 4, R.type());
    T(cv::Rect(0, 0, 3, 3)) = R * 1.0;
    T.col(3).rowRange(0, 3) = t * 1.0;

    // print R and t:
    std::cout << "Rotation matrix: " << std::endl;
    std::cout << R << std::endl;
    std::cout << "Translation vector: " << std::endl;
    std::cout << t << std::endl;

    // Reconstruct 3D points (triangulation)
    cv::Mat P0 = m_params.K * cv::Mat::eye(3, 4, CV_64F);
    cv::Mat Rt, X;
    cv::hconcat(R, t, Rt);
    cv::Mat P1 = m_params.K * Rt;
    cv::triangulatePoints(P0, P1, kpts_ref_undistorted, kpts_curr_undistorted,
                          X);
    // print rows and cols of X:
    std::cout << "X: " << X.rows << " " << X.cols << std::endl;

    // Homogeous to cartesian:
    X.row(0) = X.row(0) / X.row(3);
    X.row(1) = X.row(1) / X.row(3);
    X.row(2) = X.row(2) / X.row(3);
    X.row(3) = 1;

    // print X:

    std::vector<cv::Point3d> vec_p3d;
    for (int i = 0; i < X.cols; i++) {
      cv::Point3d p3d;
      p3d.x = X.at<double>(0, i);
      p3d.y = X.at<double>(1, i);
      p3d.z = X.at<double>(2, i);
      vec_p3d.push_back(p3d);
      // print p3d:
      if (mask.at<uchar>(0, i) == 1) {
        std::cout << "3D point: " << p3d.x << " " << p3d.y << " " << p3d.z
                  << std::endl;
      }

    }
  }

  return false;
}

}  // namespace MonocularVO

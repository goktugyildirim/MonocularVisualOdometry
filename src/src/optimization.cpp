#include "optimization.hpp"

namespace MonocularVO
{

MonocularVO::Optimization::LocalObservations
Optimization::solve_local_ba(MonocularVO::LocalObservations& local_observations,
                             const cv::Mat& K)
{
  // Load 2D points observed from multiple views:
  std::vector<std::vector<cv::Point2d>> xs = local_observations.points2D;
  std::vector<cv::Point3d> Xs = local_observations.points3D;
  std::vector<cv::Vec6d> cameras = local_observations.camera_poses;
  int count_point3D = Xs.size();

/*
  cv::Mat K;
  std::vector<double> camera_mat = {1761.2, 0, 936.1, 0, 1756.2, 539.3, 0, 0, 1};
  cv::Mat(3, 3, CV_64F, &camera_mat[0]).copyTo(K);
*/

  Eigen::Matrix<double,3,3> K_;
  K_  << 1761.2, 0, 936.1, 0, 1756.2, 539.3, 0, 0, 1;


  // Optimize camera pose and 3D points together (bundle adjustment)
  ceres::Problem ba;
  std::vector<double*> optimized_cam_poses;
  std::vector<double*> optimized_points3D;

  for (size_t j = 0; j < xs.size(); j++) //  Frames
  {
    double *camera;
    for (size_t i = 0; i < xs[j].size(); i++) // points
    {
      ceres::CostFunction *cost_func = ReprojectionError::create(xs[j][i], K_);
      camera = (double *) (&(cameras[j]));
      double *X = (double *) (&(Xs[i]));
      ceres::LossFunction* loss_func = NULL;
      loss_func = new ceres::HuberLoss(1.0);
      loss_func = new ceres::CauchyLoss(2.0);
      ba.AddResidualBlock(cost_func, loss_func, camera, X);
      if (j == 0)
        optimized_points3D.push_back(X);
    }
    optimized_cam_poses.push_back(camera);
  }
  ceres::Solver::Options options;
/*
  options.check_gradients = true;
  options.minimizer_type = ceres::MinimizerType::TRUST_REGION;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.preconditioner_type = ceres::PreconditionerType::JACOBI;
  options.visibility_clustering_type = ceres::VisibilityClusteringType::CANONICAL_VIEWS;
  options.sparse_linear_algebra_library_type = ceres::SparseLinearAlgebraLibraryType::SUITE_SPARSE;
  options.dense_linear_algebra_library_type = ceres::DenseLinearAlgebraLibraryType::LAPACK;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.trust_region_strategy_type = ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT;
  options.minimizer_progress_to_stdout = false;
*/

/*  options.gradient_tolerance = 1e-10;
  options.function_tolerance = 1e-6;
  options.parameter_tolerance = 1e-8;
  options.use_nonmonotonic_steps = true;
  options.use_inner_iterations = true;*/


  options.max_num_iterations = 12;
  options.num_threads = 12;
  //options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &ba, &summary);
  //std::cout << summary.FullReport() << std::endl;

  std::vector<cv::Vec6d> optimized_cam_poses_;
  for(const auto& c:optimized_cam_poses)
  {
    cv::Vec6d cam_pose = {*c, *(c+1), *(c+2), *(c+3), *(c+4), *(c+5)};
    //std::cout << *c << " " <<  *(c+1) << " " <<  *(c+2) << " " <<  *(c+3) << " " <<  *(c+4) << " " <<  *(c+5) << std::endl;
    optimized_cam_poses_.push_back(cam_pose);
  }

  std::vector<cv::Point3d> optimized_points3D_;
  for(const auto& c:optimized_points3D)
  {
    cv::Point3d point3D = {*c, *(c+1), *(c+2)};
    optimized_points3D_.push_back(point3D);
  }

  LocalObservations local_observations1;
  local_observations1.camera_poses = optimized_cam_poses_;
  local_observations1.points3D = optimized_points3D_;
  return local_observations1;
}

// eof solve



}
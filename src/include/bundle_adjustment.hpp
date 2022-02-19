#ifndef __BUNDLE_ADJUSTMENT__
#define __BUNDLE_ADJUSTMENT__

#include "opencv2/opencv.hpp"
#include "ceres/ceres.h"
#include "ceres/rotation.h"

// Reprojection error for bundle adjustment
struct ReprojectionError
{
    ReprojectionError(const cv::Point2d& _x, const Eigen::Matrix<double,3,3>& _K) : x(_x),  K(_K)
    {

    }

    template <typename T>
    bool operator()(const T* const camera, const T* const point, T* residuals) const
    {
        T X[3];
        ceres::AngleAxisRotatePoint(camera, point, X);
        X[0] += camera[3];
        X[1] += camera[4];
        X[2] += camera[5];

        Eigen::Matrix<T,3,1> point3d_cam = {X[0], X[1], X[2]};
        Eigen::Matrix<T,3,1>  point_projected = K*point3d_cam;

        T x_p = point_projected(0,0) / point_projected(2,0);
        T y_p = point_projected(1,0) / point_projected(2,0);
        // residual = x - x'
        residuals[0] = T(x.x) - x_p;
        residuals[1] = T(x.y) - y_p;
        return true;
    }

    static ceres::CostFunction* create(const cv::Point2d& _x, const Eigen::Matrix<double,3,3>& _K)
    {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(new ReprojectionError(_x, _K)));
    }

private:
    const Eigen::Matrix<double,3,3> K;
    const cv::Point2d x;
};

#endif // End of '__BUNDLE_ADJUSTMENT__'

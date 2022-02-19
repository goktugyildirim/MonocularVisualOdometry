#ifndef BUILD_SRC_INCLUDE_PARAMS_HPP_
#define BUILD_SRC_INCLUDE_PARAMS_HPP_

#include <opencv2/core.hpp>
#include <string>

namespace BundleAdjustment
{

  struct Params
  {
    int max_frame_count_to_key_frame = 15;
    double max_angular_px_disp = 500;
    float ransac_outlier_threshold = 1;

    // Key-point extraction params:
    bool use_modern = true;
    std::string detector_type = "SIFT"; // modern: ORB FAST AKAZE SIFT BRISK | classic: SHITOMASI HARRIS
    std::string descriptor_type = "SIFT"; // BRIEF, ORB, FREAK, AKAZE, SIFT
    std::string descriptor_type_ = "DES_HOG"; // DES_BINARY, DES_HOG
    std::string  matcher_type = "MAT_BF";        // MAT_BF, MAT_FLANN
    std::string  selector_type = "SEL_KNN";       // SEL_NN, SEL_KNN

    int max_orb_detect = 1000;
    float max_desc_dist = 70; // keyframe matching distance threshold
    int count_nonmax_keep = 200;
    int count_min_tracked = 100; // threshold for new key frame selecting
    int fast_threshold = 100; // if it is increased, hz is increased
    int shitomasi_block_size = 25;

    std::vector<double> dist_coeff = {0.066343, -0.14401, 0.00283, -0.003766, -0.025824};
    std::vector<double> camera_mat = {1761.2, 0, 936.1,
                                      0, 1756.2, 539.3,
                                      0, 0, 1};

    cv::Mat K = cv::Mat::eye(3,3,6);
    cv::Mat mat_dist_coeff;

    Params() = default;
    Params(const bool& use_modern,
           const std::string& detector_type,
           const std::string& descriptor_type,
           const std::string&  matcher_type,
           const std::string&  selector_type,
           const int& max_orb_detect,
           const float& max_desc_dist,
           const int& count_nonmax_keep,
           const int& fast_threshold,
           const int& count_min_tracked,
           const int& shitomasi_block_size,
           const int& max_frame_count_to_key_frame,
           const double& max_angular_px_disp,
           const float& ransac_outlier_threshold)
    {
      this->use_modern = use_modern;
      this->detector_type = detector_type;
      this->descriptor_type = descriptor_type;
      this->matcher_type = matcher_type;
      this->selector_type = selector_type;

      if (this->descriptor_type == "SIFT")
      {
        this->descriptor_type_ = "DES_HOG";
      }
      else
      {
        this->descriptor_type_ = "DES_BINARY";
      }

      cv::Mat(1, 5, 6, &dist_coeff[0]).copyTo(mat_dist_coeff);
      cv::Mat(3, 3, 6, &camera_mat[0]).copyTo(K);

      this->max_orb_detect = max_orb_detect;
      this->max_desc_dist = max_desc_dist;
      this->count_nonmax_keep = count_nonmax_keep;
      this->count_min_tracked = count_min_tracked;
      this->fast_threshold = fast_threshold;
      this->shitomasi_block_size = shitomasi_block_size;
      this->max_frame_count_to_key_frame = max_frame_count_to_key_frame;
      this->max_angular_px_disp = max_angular_px_disp;
      this->ransac_outlier_threshold = ransac_outlier_threshold;
    }
  };


}

#endif  // BUILD_SRC_INCLUDE_PARAMS_HPP_

#include "vision.hpp"

using namespace std::chrono;

namespace MonocularVO
{

void
MonocularVO::Vision::adaptiveNonMaximalSuppresion(
    std::vector<cv::KeyPoint>& keypoints,
    const int& numToKeep)
{
  if( keypoints.size() < numToKeep ) { return; }

  //
  // Sort by response
  //
  std::sort( keypoints.begin(), keypoints.end(),
            [&]( const cv::KeyPoint& lhs, const cv::KeyPoint& rhs )
            {
              return lhs.response > rhs.response;
            } );

  std::vector<cv::KeyPoint> anmsPts;

  std::vector<double> radii;
  radii.resize( keypoints.size() );
  std::vector<double> radiiSorted;
  radiiSorted.resize( keypoints.size() );

  const float robustCoeff = 1.11; // see paper

  for( int i = 0; i < keypoints.size(); ++i )
  {
    const float response = keypoints[i].response * robustCoeff;
    double radius = std::numeric_limits<double>::max();
    for( int j = 0; j < i && keypoints[j].response > response; ++j )
    {
      radius = std::min( radius, cv::norm( keypoints[i].pt - keypoints[j].pt ) );
    }
    radii[i]       = radius;
    radiiSorted[i] = radius;
  }

  std::sort(radiiSorted.begin(), radiiSorted.end(),
            [&]( const double& lhs, const double& rhs )
            {
              return lhs > rhs;
            } );

  const double decisionRadius = radiiSorted[numToKeep];
  for( int i = 0; i < radii.size(); ++i )
  {
    if( radii[i] >= decisionRadius )
    {
      anmsPts.push_back( keypoints[i] );
    }
  }
  anmsPts.swap( keypoints );
}



void
MonocularVO::Vision::detect_keypoints(
  const MonocularVO::Params& params,
  std::vector<cv::KeyPoint> &keypoints,
  cv::Mat& img)
{
  if(params.use_modern)
  {
    Vision::keypoints_modern(params,keypoints,img, params.detector_type);
  } else
  {
    if (params.detector_type == "SHITOMASI")
      Vision::keypoints_shitomasi(keypoints, img, params);
    else if (params.detector_type == "HARRIS")
      Vision::keypoints_harris(keypoints, img);
  }
}

void
MonocularVO::Vision::keypoints_modern(
  const MonocularVO::Params& params,
  std::vector<cv::KeyPoint> &keypoints,
  cv::Mat &img,
  const std::string &detectorType)
{
  cv::Ptr<cv::FeatureDetector> feature_detector;

  if (detectorType == "ORB")
    feature_detector = cv::ORB::create(params.max_orb_detect, 1.2, 15,
     31, 0, 2, cv::ORB::HARRIS_SCORE, 50, 100);
  else if (detectorType == "FAST")
    feature_detector = cv::FastFeatureDetector::create(
        params.fast_threshold, true,
        cv::FastFeatureDetector::TYPE_9_16);
  else if (detectorType == "AKAZE")
    feature_detector = cv::AKAZE::create();
  else if (detectorType == "SIFT")
    feature_detector = cv::SIFT::create(300);
  else if (detectorType == "BRISK")
    feature_detector = cv::BRISK::create();
  else
    std::cout << "Please use valid detector name." << std::endl;

  feature_detector->detect(img, keypoints);
}


void
MonocularVO::Vision::desc_keypoints(
    const MonocularVO::Params& params,
    std::vector<cv::KeyPoint> &keypoints,
    cv::Mat& descriptors,
    const cv::Mat& img)
{
// select appropriate descriptor
  cv::Ptr<cv::DescriptorExtractor> extractor;
  if (params.descriptor_type=="BRISK") {
    int threshold = 30;        // FAST/AGAST detection threshold score.
    int octaves = 3;           // detection octaves (use 0 to do single scale)
    float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.
    extractor = cv::BRISK::create(threshold, octaves, patternScale);
  }
  else if (params.descriptor_type=="AKAZE")
    extractor = cv::AKAZE::create();
  else if (params.descriptor_type=="SIFT")
    extractor = cv::SIFT::create(300);
  else if (params.descriptor_type=="ORB")
    extractor = cv::ORB::create(params.max_orb_detect);
  else if (params.descriptor_type=="FREAK")
    extractor = cv::xfeatures2d::FREAK::create();
  else if (params.descriptor_type=="BRIEF")
    extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
  else if (params.descriptor_type=="BEBLID")
    extractor = cv::xfeatures2d::BEBLID::create(0.1);
  else
    std::cout << "Please use valid descriptor name." << std::endl;

  extractor->compute(img, keypoints, descriptors);
}

void MonocularVO::Vision::keypoints_shitomasi(
  std::vector<cv::KeyPoint> &keypoints,
  cv::Mat &img,
  const MonocularVO::Params& params)
{
// compute detector parameters based on image size
  int blockSize =
     params.shitomasi_block_size;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
  double maxOverlap = 0.1; // max. permissible overlap between two features in %
  double minDistance = (1.0 - maxOverlap) * blockSize;
  int maxCorners = img.rows * img.cols / std::max(1.0, minDistance); // max. num. of keypoints

  double qualityLevel = 0.01; // minimal accepted quality of image corners
  double k = 0.04;

  // Apply corner detection
  std::vector<cv::Point2f> corners;
  cv::goodFeaturesToTrack(img, corners, 99999999, qualityLevel,
        minDistance, cv::Mat(), blockSize, true, k);

  // add corners to result vector
  for (auto it = corners.begin(); it != corners.end(); ++it) {

    cv::KeyPoint newKeyPoint;
    newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
    newKeyPoint.size = blockSize;
    keypoints.push_back(newKeyPoint);
  }
}

void
MonocularVO::Vision::keypoints_harris(
  std::vector<cv::KeyPoint> &keypoints,
  cv::Mat &img)
{
  // Detector parameters
  int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
  int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
  int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
  double k = 0.04;       // Harris parameter (see equation for details)

  // Detect Harris corners and normalize output
  cv::Mat dst, dst_norm, dst_norm_scaled;
  dst = cv::Mat::zeros(img.size(), CV_32FC1);
  cv::cornerHarris(img, dst, blockSize, apertureSize,
      k, cv::BORDER_DEFAULT);
  cv::normalize(dst, dst_norm, 0, 255,
      cv::NORM_MINMAX, CV_32FC1, cv::Mat());
  cv::convertScaleAbs(dst_norm, dst_norm_scaled);

  // Do non-max suppression:
  double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
  for (size_t j = 0; j < dst_norm.rows; j++) {
    for (size_t i = 0; i < dst_norm.cols; i++) {
      int response = (int) dst_norm.at<float>(j, i);
      if (response > minResponse) { // only store points above a threshold

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f(i, j);
        newKeyPoint.size = 2 * apertureSize;
        newKeyPoint.response = response;

        // perform non-maximum suppression (NMS) in local neighbourhood around new key point
        bool bOverlap = false;
        for (auto it = keypoints.begin(); it != keypoints.end(); ++it) {
          double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
          if (kptOverlap > maxOverlap) {
            bOverlap = true;
            if (newKeyPoint.response
                > (*it).response) { // if overlap is >t AND response is higher for new kpt
              *it = newKeyPoint; // replace old key point with new one
              break;             // quit loop over keypoints
            }
          }
        }
        if (!bOverlap) {  // only add new key point if no overlap has been found in previous NMS
          keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
        }
      }
    } // eof loop over cols
  }     // eof loop over rows
}

void
MonocularVO::Vision::match_descriptors(
   std::vector<cv::KeyPoint> &kpts_source,
   std::vector<cv::KeyPoint> &kpts_ref,
   cv::Mat &desc_source,
   cv::Mat &desc_ref,
   std::vector<cv::DMatch> &matched_kpts,
   const MonocularVO::Params& params)
{
  // configure matcher
  bool crossCheck = false;
  cv::Ptr<cv::DescriptorMatcher> matcher;

  if (params.matcher_type == "MAT_BF")
  {
    int normType = params.descriptor_type_ ==
        "DES_HOG" ? cv::NORM_L2 : cv::NORM_HAMMING;
    matcher = cv::BFMatcher::create(normType, crossCheck);
  }
  else if (params.matcher_type == "MAT_FLANN")
  {
    // add this condition due to error: error:
    // (-210:Unsupported format or combination of formats) in function 'buildIndex_'
    if (desc_source.type() != CV_32F || desc_ref.type() != CV_32F)
    {
      desc_source.convertTo(desc_source, CV_32F);
      desc_ref.convertTo(desc_ref, CV_32F);
    }
    matcher = cv::DescriptorMatcher::create(
        cv::DescriptorMatcher::FLANNBASED);
  }
  else if (params.matcher_type == "BruteForce-Hamming")
  {
    matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
  }

  // perform matching task
  if (params.selector_type == "SEL_NN")
  { // nearest neighbor (best match)

    matcher->match(desc_source,
                   desc_ref,
                   matched_kpts);

  }

  else if (params.selector_type == "SEL_KNN")
  { // k nearest neighbors (k=2)
    std::vector<std::vector<cv::DMatch>> vector_matches;
    matcher->knnMatch(desc_source, desc_ref,
                      vector_matches, 2);

    // descriptor distance filtering
    double min_ratio = 0.8;
    for (size_t i=0; i<vector_matches.size(); i++)
    {
      auto match = vector_matches[i];
      if (match[0].distance < min_ratio * match[1].distance)
      {
        if (match[0].distance < params.max_desc_dist)
          matched_kpts.push_back(match[0]);
      }

    }
  }
}




void
MonocularVO::Vision::draw_keypoints(
  cv::Mat &image, const std::vector<cv::KeyPoint> &keypoints)
{
  for (const auto& kpt:keypoints)
    cv::circle(image, kpt.pt, 1,
     cv::Scalar(255, 0, 0), 3, 4, 0);
}


void
MonocularVO::Vision::make_img_3_channel(
  cv::Mat &img)
{
  std::vector<cv::Mat> copies{img,img,img};
  cv::merge(copies,img);
}


cv::Mat
MonocularVO::Vision::get_F(
    FrameSharedPtr& view1,
    FrameSharedPtr& view2,
    const float& ransac_threshold,
    cv::Mat& inliers_F)
{
  cv::Mat F = cv::findFundamentalMat(
    view1->keypoints_p2d,
    view2->keypoints_p2d,
    cv::FM_RANSAC,
    ransac_threshold, 0.99, 500,
    inliers_F);
  return F;
}


cv::Mat
MonocularVO::Vision::get_E(
        MonocularVO::FrameSharedPtr& view1,
        MonocularVO::FrameSharedPtr& view2,
        const float& ransac_threshold,
        cv::Mat& inliers_E,
        const cv::Mat& K)
{
  cv::Mat E = cv::findEssentialMat(view1->keypoints_p2d,
         view2->keypoints_p2d, K, cv::RANSAC, 0.99,
         ransac_threshold, inliers_E);
  return E;
}






void
MonocularVO::Vision::extract_features(
   FrameSharedPtr& frame,
   const MonocularVO::Params& params)
{
  auto start = std::chrono::steady_clock::now();
  frame->keypoints_p2d.clear();
  Vision::detect_keypoints(params, frame->keypoints, frame->image_gray);
  Vision::desc_keypoints(params, frame->keypoints, frame->descriptors,
                         frame->image_gray);
  //std::cout << "Detected keypoint count: " << frame->keypoints.size() << std::endl;

  // Optical flow tracker takes vector<Point2f> as an input
  for (const auto& keypoint: frame->keypoints)
    frame->keypoints_p2d.push_back(keypoint.pt);

  frame->is_feature_extracted = true;
  auto end = std::chrono::steady_clock::now();
  std::cout << "Keypoint extraction tooks: "
       << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
       << " millisecond." << std::endl;
}



double
Vision::average_ang_px_displacement(
  const std::vector<cv::Point2f> &prev_frame,
  const std::vector<cv::Point2f> &curr_frame,
  const float& img_height, const float& img_width)
{
  double average_displacement = 0;
  assert(prev_frame.size() == curr_frame.size());
  for (int i=0; i< prev_frame.size(); i++)
  {
    float delta_height = (img_height - prev_frame[i].y) + curr_frame[i].y;
    float delta_width = abs(prev_frame[i].x-curr_frame[i].x);
    double delta_degree = abs(atan2(delta_width, delta_height))*57.29577;
    average_displacement += delta_degree;
  }
  return average_displacement/prev_frame.size();
}


cv::Point2f
Vision::pixel_2_cam_norm_plane(const cv::Point2f &p,
                                           const cv::Mat &K)
{
  return cv::Point2f(
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

cv::Point3f
Vision::pixel_2_cam(
  const cv::Point2f &p, const cv::Mat &K,
  double depth)
{
  return cv::Point3f(
      depth * (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      depth * (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1),
      depth);
}


cv::Point2f
Vision::cam_2_pixel(const cv::Point3f &p, const cv::Mat &K)
{
  return cv::Point2f(
      K.at<double>(0, 0) * p.x / p.z + K.at<double>(0, 2),
      K.at<double>(1, 1) * p.y / p.z + K.at<double>(1, 2));
}


void Vision::pose_estimation_2d2d(
  const std::vector<cv::Point2f> &kpts_prev_frame,
  const std::vector<cv::Point2f> &kpts_curr_frame,
  const MonocularVO::Params& params,
  cv::Mat& R, cv::Mat& t)
{
  double fx = params.K.at<double>(0,0);
  double fy = params.K.at<double>(1,1);
  double cx = params.K.at<double>(0,2);
  double cy = params.K.at<double>(1,2);
  double f =  params.K.at<double>(1,1);
  cv::Point2d c(cx, cy);
  // Undistorted points:
  // cv::undistortPoints(kpts_ref, kpts_ref_undistorted_norm_inlier,
  // m_params.K, m_params.mat_dist_coeff, cv::noArray(), m_params.K);
  std::vector<cv::Point2f> kpts_ref_undistorted_norm;
  std::vector<cv::Point2f> kpts_curr_undistorted_norm;
  cv::undistortPoints(kpts_prev_frame, kpts_ref_undistorted_norm, params.K,
                      params.mat_dist_coeff);
  cv::undistortPoints(kpts_curr_frame, kpts_curr_undistorted_norm, params.K,
                      params.mat_dist_coeff);
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
                                     0.2,0.99,
                                     inlier_mask);
  cv::Mat E = params.K.t() * F * params.K;
  // Essential matrix cannot zero matrix:
  std::cout << "E = " << std::endl << E << std::endl;

  int count_inlier = cv::recoverPose(
      E, kpts_ref_undistorted_focal,
      kpts_curr_undistorted_focal, R, t);

  std::cout << "Inlier count: " << count_inlier << std::endl;

  cv::Mat t_x =
      (cv::Mat_<double>(3, 3) << 0, -t.at<double>(2, 0), t.at<double>(1, 0),
       t.at<double>(2, 0), 0, -t.at<double>(0, 0),
       -t.at<double>(1, 0), t.at<double>(0, 0), 0);

  for (int i=0; i<kpts_ref_undistorted_norm.size(); i++)
  {
    cv::Point2d pt1 = kpts_ref_undistorted_norm[i];
    cv::Mat y1 = (cv::Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
    cv::Point2d pt2 = kpts_curr_undistorted_norm[i];
    cv::Mat y2 = (cv::Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
    cv::Mat d = y2.t() * t_x * R * y1;
    //std::cout << "epipolar constraint = " << d << std::endl;
  }
}

} // eof MonocularVO
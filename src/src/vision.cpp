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
    feature_detector = cv::ORB::create(params.max_orb_detect);
  else if (detectorType == "FAST")
    feature_detector = cv::FastFeatureDetector::create(
        params.fast_threshold, true,
        cv::FastFeatureDetector::TYPE_9_16);
  else if (detectorType == "AKAZE")
    feature_detector = cv::AKAZE::create();
  else if (detectorType == "SIFT")
    feature_detector = cv::SIFT::create();
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
    extractor = cv::SIFT::create();
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
  double maxOverlap = 0.0; // max. permissible overlap between two features in %
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
MonocularVO::Vision::refine_matches(
  FrameSharedPtr &view1,
  FrameSharedPtr &view2,
  const cv::Mat &inliers_F,
  const cv::Mat &inliers_E,
  std::vector<int> &old_kpt_ids,
  std::vector<int> &new_kpt_ids)
{
  int indexCorrection = 0;
  for(size_t i=0; i<inliers_E.rows; i++)
  {
    if (inliers_F.at<bool>(i,0) == false or inliers_E.at<bool>(i,0) == false)
    {
      old_kpt_ids.erase(old_kpt_ids.begin() + i - indexCorrection);
      new_kpt_ids.erase(new_kpt_ids.begin() + i - indexCorrection);
      view1->keypoints_p2d.erase (view1->keypoints_p2d.begin() + i - indexCorrection);
      view2->keypoints_p2d.erase (view2->keypoints_p2d.begin() + i - indexCorrection);
      indexCorrection++;
    }
  }

}



void
MonocularVO::Vision::recover_pose(
  const MonocularVO::FrameSharedPtr& view1,
  const MonocularVO::FrameSharedPtr& view2,
  const cv::Mat& F,
  const cv::Mat& E,
  cv::Mat& R,
  cv::Mat& t,
  const cv::Mat& K)
{
  cv::Point2d c = {K.at<double>(0, 3),
                   K.at<double>(1,3)};
  double focal_l = K.at<double>(0,0);
  cv::Mat inliers;
  int inlier_num = cv::recoverPose(
      E, view1->keypoints_p2d,
      view2->keypoints_p2d,
      R, t, focal_l, c);

/*  int c = 0;
  for(size_t i=0; i<inliers.rows; i++)
  {
    if (inliers.at<bool>(i,0) == false)
    {
      c++;
    }
  }
  std::cout << "recovery pose outliers count: " << c++ << std::endl;*/

}



void
MonocularVO::Vision::extract_features(
   FrameSharedPtr& frame,
   std::vector<cv::Point2f>& m_ref_keypoints,
   const MonocularVO::Params& params)
{
  auto start = std::chrono::steady_clock::now();
  m_ref_keypoints.clear();
  frame->keypoints_p2d.clear();
  Vision::detect_keypoints(params, frame->keypoints, frame->image_gray);
  Vision::desc_keypoints(params, frame->keypoints, frame->descriptors,
                         frame->image_gray);
  Vision::adaptiveNonMaximalSuppresion(frame->keypoints,2000);
  for (const auto& keypoint: frame->keypoints)
  {
    frame->keypoints_p2d.push_back(keypoint.pt);
    m_ref_keypoints.push_back(keypoint.pt);
  }
  frame->is_feature_extracted = true;

  auto end = std::chrono::steady_clock::now();
  std::cout << "Feature extraction tooks: "
       << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
       << " millisecond." << std::endl;
}

/*

int
MonocularVO::Vision::track_features(
  const MonocularVO::Params& params,
  MapInitialSharedPtr & map,
  cv::Mat& R,
  cv::Mat& t)
{
  FrameSharedPtr prev_frame = map->get_prev_frame();
  FrameSharedPtr curr_frame = map->get_curr_frame();

// calculate optical flow
  std::vector<uchar> status;
  std::vector<float> err;
  cv::Size winSize = cv::Size(25,25);
  cv::TermCriteria termcrit=cv::TermCriteria(
  cv::TermCriteria::COUNT+cv::TermCriteria::EPS,
  30, 0.01);
*/
/*
  std::cout << "Before KLT between | previous frame " << prev_frame->frame_id << " kpt count: "
  << prev_frame->keypoints_p2d.size() <<  " | current frame " << curr_frame->frame_id <<
  " kpt count: " << curr_frame->keypoints_p2d.size() << std::endl;
*//*

  cv::calcOpticalFlowPyrLK(prev_frame->image_gray,
                           curr_frame->image_gray,
                           prev_frame->keypoints_p2d,
                           curr_frame->keypoints_p2d,
                           status, err, winSize,
                           3, termcrit, 0, 0.001);

  assert(prev_frame->keypoints_p2d.size() == curr_frame->keypoints_p2d.size());

  map->update_past_frames_optical_flow(status);
*/
/*
  std::cout << "After KLT between | previous frame " << prev_frame->frame_id << " kpt count: "
            << prev_frame->keypoints_p2d.size() <<  " | current frame " << curr_frame->frame_id <<
            " kpt count: " << curr_frame->keypoints_p2d.size() << std::endl;
*//*


  // Calculate fundamental matrix
  cv::Mat inliers_F;
  cv::Mat F = Vision::get_F(prev_frame, curr_frame,
            params.ransac_outlier_threshold, inliers_F);
  // Calculate essential matrix
  cv::Mat inliers_E;
  cv::Mat E = Vision::get_E(prev_frame, curr_frame,
            params.ransac_outlier_threshold, inliers_E, params.K);
  //E = params.K.t() * F * params.K;

  assert(inliers_F.rows == inliers_E.rows == prev_frame->keypoints_p2d.size());

  // Refine matches
  map->update_past_frames_epipolar(inliers_F, inliers_E);

  assert(prev_frame->keypoints_p2d.size() == curr_frame->keypoints_p2d.size());

  */
/*std::cout << "After match refinement | prev " << prev_frame->frame_id << " and curr keyframe " <<
            curr_frame->frame_id << " | matched kpt cout: " << curr_frame->keypoints_p2d.size() << std::endl;*//*



  Vision::recover_pose(prev_frame, curr_frame, F, E, R, t, params.K);

  int count_local_landmark_ = map->count_local_landmark_;
  return count_local_landmark_;
}

*/



MonocularVO::Vision::MatchKeyFrameSharedPtr
MonocularVO::Vision::match_key_frames(
  const MonocularVO::Params& params,
  FrameSharedPtr& keyframe_old,
  FrameSharedPtr& keyframe_new)
{
  std::cout << "Before matching keyframe worked | prev key-frame kpt count: "
  << keyframe_old->keypoints.size() << "| curr key-frame kpt count: " <<
  keyframe_new->keypoints.size() <<  std::endl;

  std::vector<cv::DMatch> matches;
  Vision::match_descriptors(keyframe_old->keypoints,
                            keyframe_new->keypoints,
                            keyframe_old->descriptors,
                            keyframe_new->descriptors,matches,params);

  keyframe_old->keypoints_p2d.clear();
  keyframe_new->keypoints_p2d.clear();

  // Left view Image(t-1) reference frame - match queryIdx
  // Right view Image(t) source frame - match trainIdx
  std::vector<int> old_kpt_ids;
  std::vector<int> new_kpt_ids;
  for (const auto &match_kpt:matches)
  {
    old_kpt_ids.push_back(match_kpt.queryIdx);
    new_kpt_ids.push_back(match_kpt.trainIdx);
    cv::Point2f kpt_left = keyframe_old->keypoints[match_kpt.queryIdx].pt;
    cv::Point2f kpt_right = keyframe_new->keypoints[match_kpt.trainIdx].pt;
    keyframe_old->keypoints_p2d.push_back(kpt_left);
    keyframe_new->keypoints_p2d.push_back(kpt_right);
  }

  std::cout << "After keypoint descriptor matching, match count: " <<
  keyframe_new->keypoints_p2d.size() << std::endl;

  // Calculate fundamental matrix
  cv::Mat inliers_F;
  cv::Mat F = Vision::get_F(keyframe_old, keyframe_new, 1.5, inliers_F);
  // Calculate essential matrix
  cv::Mat inliers_E;
  cv::Mat E = Vision::get_E(keyframe_old, keyframe_new, 1.5, inliers_E,
                             params.K);

  // Refine matches
  Vision::refine_matches(keyframe_new, keyframe_old, inliers_F, inliers_E,
                         old_kpt_ids, new_kpt_ids);

  std::cout << "After refinement match count: " << keyframe_new->keypoints_p2d.size() << std::endl;

  assert(keyframe_old->keypoints_p2d.size() == keyframe_new->keypoints_p2d.size()
             == old_kpt_ids.size() == new_kpt_ids.size());


  MatchKeyFrameSharedPtr match_key_frame = std::make_shared<MatchKeyFrame>(keyframe_old->frame_id,
                                                                           keyframe_old->keypoints,
                                                                           old_kpt_ids,
                                                                           keyframe_new->frame_id,
                                                                           keyframe_new->keypoints,
                                                                           new_kpt_ids);
  return match_key_frame;





  /*std::cout << "Old frame kpt count: " << keyframe_old->keypoints.size() << std::endl;

  Vision::extract_keyframe_features(keyframe_new, params);

  // End of matching assign again for optical flow
  std::vector<cv::Point2f> new_kpts_tmp;
  std::vector<cv::Point2f> new_kpts_tmp;
  for (const auto& p: keyframe_new->keypoints)
    new_kpts_tmp.push_back(p.pt);

  std::cout << "New frame kpt count: " << keyframe_new->keypoints.size() << std::endl;

  std::vector<cv::DMatch> matches;
  Vision::match_descriptors(keyframe_old->keypoints,
                            keyframe_new->keypoints,
                            keyframe_old->descriptors,
                            keyframe_new->descriptors,
                            matches,
                            params);

  // Clear old keyframe kpts:
  keyframe_old->keypoints_p2d.clear();
  keyframe_new->keypoints_p2d.clear();

  // Left view Image(t-1) reference frame - match queryIdx
  // Right view Image(t) source frame - match trainIdx
  std::vector<int> old_kpt_ids;
  std::vector<int> new_kpt_ids;
  for (const auto &match_kpt:matches)
  {
    old_kpt_ids.push_back(match_kpt.queryIdx);
    new_kpt_ids.push_back(match_kpt.trainIdx);
    cv::Point2f kpt_left = keyframe_old->keypoints[match_kpt.queryIdx].pt;
    cv::Point2f kpt_right = keyframe_new->keypoints[match_kpt.trainIdx].pt;
    keyframe_old->keypoints_p2d.push_back(kpt_left);
    keyframe_new->keypoints_p2d.push_back(kpt_right);
  }

  std::cout << "After keypoint descriptor matching, match count: " << keyframe_new->keypoints_p2d.size() << std::endl;

  // Calculate fundamental matrix
  cv::Mat inliers_F;
  cv::Mat F = Vision::get_F(keyframe_old, keyframe_new, 1, inliers_F);
  // Calculate essential matrix
  cv::Mat inliers_E;
  cv::Mat E = Vision::get_E(keyframe_old, keyframe_new, 1, inliers_E);

  // Refine matches
  Vision::refine_matches(keyframe_new, keyframe_old, inliers_F, inliers_E,
                         old_kpt_ids, new_kpt_ids);

  assert(keyframe_old->keypoints_p2d.size() == keyframe_new->keypoints_p2d.size()
  == old_kpt_ids.size() == new_kpt_ids.size());

  std::cout << "After refinement match count: " << keyframe_new->keypoints_p2d.size() << std::endl;

  cv::Mat img_concat = Vision::draw_matches(keyframe_old, keyframe_new);

  // Due to optical flow
  keyframe_new->keypoints_p2d = new_kpts_tmp;

  MatchKeyFrameSharedPtr match_key_frame = std::make_shared<MatchKeyFrame>(keyframe_old->frame_id,
                                                                           keyframe_old->keypoints,
                                                                           old_kpt_ids,
                                                                           keyframe_new->frame_id,
                                                                           keyframe_new->keypoints,
                                                                           new_kpt_ids,
                                                                           img_concat);
  return match_key_frame;*/
}



/*cv::Mat
Vision::visualize_feature_tracking(
    const MapInitialSharedPtr& map,
    const bool& save_images,
    const bool& draw_line)
{
  cv::Mat concat_img;

  FrameSharedPtr last_key_frame = *(map->frames_.begin()+map->get_last_key_frame_id_in_frames());
  cv::Mat img_last_key_frame = last_key_frame->image_gray_with_kpts;

  FrameSharedPtr last_frame = *(map->frames_.end()-1);
  cv::Mat img_last_frame = last_frame->image_gray_with_kpts;

*//*  std::cout << "Matches between frame ids are visualized: " << last_key_frame->frame_id << " " <<
  last_frame->frame_id << std::endl;*//*
  cv::Mat img_concat;
  cv::vconcat(img_last_key_frame,
              img_last_frame,
              concat_img);

  for (int i=0; i<last_frame->keypoints_p2d.size(); i++)
  {
    cv::Point2f px_upper = last_key_frame->keypoints_p2d[i];
    cv::Point2f px_lower = last_frame->keypoints_p2d[i];
    px_lower.y += last_key_frame->image_gray.rows;
    if (draw_line)
    {
      cv::line(concat_img, px_upper, px_lower,
         cv::Scalar(0, 255, 0),
         2, cv::LINE_8);
    }

    cv::circle(concat_img, last_key_frame->keypoints_p2d[i], 3,
               cv::Scalar(0, 0, 255),
               2, 4, 0);

    cv::Point2f p = last_frame->keypoints_p2d[i];
    p.y += last_key_frame->image_gray.rows;;
    cv::circle(concat_img, p, 4,
               cv::Scalar(0, 0, 255),
               3, 4, 0);
  }

  // Save images:
  if (save_images)
  {
    int counter_img = 0;
    std::string path = "/home/goktug/Desktop/test_images/";
    for (auto it=map->frames_.begin()+map->get_last_key_frame_id_in_frames();
         it!=map->frames_.end(); it++)
    {
      cv::Mat img;
      (*it)->image_gray_with_kpts.copyTo(img);
      for (int i=0; i<(*it)->keypoints_p2d.size(); i++)
      {
        cv::circle(img, (*it)->keypoints_p2d[i], 1,
                   cv::Scalar(0, 0, 255),
                   2, 4, 0);
      }
      std::string curr_path = path + std::to_string(counter_img) + ".jpg";
      cv::imwrite(curr_path, img);
      counter_img++;
    }
  }

  if (concat_img.empty())
   throw std::runtime_error("Concat img is empty!");
*//*

  last_frame->image_gray.release();
  last_frame->image_gray_with_kpts.release();
  last_frame->img_colored.release();
*//*

  return concat_img;
}*/


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

} // eof MonocularVO
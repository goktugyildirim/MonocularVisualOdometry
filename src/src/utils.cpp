#include "utils.hpp"

namespace MonocularVO
{

void
Utils::publish_image(
    const rclcpp::Publisher<ImageMsgT>::SharedPtr &pub_img,
    const cv::Mat &image,
    const rclcpp::Time &time,
    const std::string &frame_id)
{
  std_msgs::msg::Header header;
  header.frame_id = frame_id;
  header.stamp = time;
  sensor_msgs::msg::Image::SharedPtr ros_image_ptr(new sensor_msgs::msg::Image());

  ros_image_ptr = cv_bridge::CvImage(header,
         sensor_msgs::image_encodings::BGR8, // MONO8 or BGR8
         image).toImageMsg();
  pub_img->publish(*ros_image_ptr);
}



cv::Mat
Utils::get_pose(
  const cv::Mat &R,
  const cv::Mat &t,
  const float &scale)
{
  cv::Mat T = cv::Mat::eye(4, 4, 6);
  T(cv::Rect(0, 0, 3, 3)) = R * 1.0;
  T.col(3).rowRange(0, 3) = t * 1.0;
  return  T;
}


cv::Point2d
Utils::project_point(
  const cv::Mat &R,
  const cv::Mat &t,
  const cv::Mat &K,
  const cv::Point3d &point_3D)
{
  cv::Point2d projected_point;

  // Define transformation matrix world to camera:
  cv::Mat pose_world_to_cam = Utils::get_pose(R, t ,1); // 4x4
  // Crete 3D point in homogeneous coordinates:
  cv::Mat point_3D_homogeneous_world_frame = cv::Mat::zeros(4, 1, 6); // 4x1
  point_3D_homogeneous_world_frame.at<double>(0,0) = point_3D.x;
  point_3D_homogeneous_world_frame.at<double>(1,0) = point_3D.y;
  point_3D_homogeneous_world_frame.at<double>(2,0) = point_3D.z;
  point_3D_homogeneous_world_frame.at<double>(3,0) = 1;

  cv::Mat mat_projector = K * pose_world_to_cam(cv::Rect(0, 0, 4, 3)); // 3x4

  cv::Mat bottom = cv::Mat::zeros(1,4, 6);
  bottom.at<double>(0,3) = 1;
  cv::Mat mat_point_transformer;
  cv::vconcat(mat_projector, bottom, mat_point_transformer);

/*  std::cout << "mat_point_transformer: \n" << mat_point_transformer <<
  "\n" << mat_point_transformer.rows << " " << mat_point_transformer.cols << std::endl;*/

  cv::Mat mat_projected_point = mat_point_transformer*point_3D_homogeneous_world_frame;
  projected_point.x = mat_projected_point.at<double>(0,0) / mat_projected_point.at<double>(2,0);
  projected_point.y = mat_projected_point.at<double>(1,0) / mat_projected_point.at<double>(2,0);

  return projected_point;
}


void
Utils::update_curr_frame_descriptor(
  const cv::Mat &matrix, const std::vector<cv::DMatch> &matches,
  cv::Mat &matrix_without_row)
{
  matrix_without_row = cv::Mat::zeros(matches.size(), matrix.cols, 6);
  int row_counter = 0;
  for (const cv::DMatch &match : matches)
  {
    matrix.row(match.trainIdx).copyTo(matrix_without_row.row(row_counter));
    row_counter++;
  }
}


void
Utils::get_not_matched_kpt_ids(
  const FrameSharedPtr &frame,
  const std::vector<cv::DMatch> &matches,
  std::vector<int> &vector_not_matched_kpt_ids,
  const bool& is_prev_frame)
{
  for (int i = 0; i < frame->keypoints.size(); i++)
  {
    bool is_matched = false;
    for (int j = 0; j < matches.size(); j++)
    {
      if (is_prev_frame) {
        if (matches[j].queryIdx == i) {
          is_matched = true;
          break;
        }
      }
      else {
        if (matches[j].trainIdx == i) {
          is_matched = true;
          break;
        }
      }
    }
    if (!is_matched)
    {
      vector_not_matched_kpt_ids.push_back(i);
    }
  }
}

void
Utils::remove_lost_landmark_ids(
  const std::vector<int> &list_of_index,
  std::vector<int> &vector_to_remove_elements)
{
  std::vector<int> cp;
  for (int i = 0; i < vector_to_remove_elements.size(); i++)
    if (std::find(list_of_index.begin(), list_of_index.end(), i) == list_of_index.end())
      cp.push_back(vector_to_remove_elements[i]);
  vector_to_remove_elements = cp;
}

void
Utils::print_vector_elements(const std::vector<int> &vector_to_print)
{
  for (int i = 0; i < vector_to_print.size(); i++)
    std::cout << vector_to_print[i] << " ";
  std::cout << std::endl;
}

void
Utils::print_keypoints_with_indexes(
  const std::vector<cv::KeyPoint> &vector_to_print)
{
  for (int i = 0; i < vector_to_print.size(); i++)
    std::cout << i << " " << vector_to_print[i].pt << " | ";
  std::cout << std::endl;
}

void
Utils::print_keypoints_p2d_with_indexes(
  const std::vector<cv::Point2f> &vector_to_print)
{
  for (int i = 0; i < vector_to_print.size(); i++)
    std::cout << i << " " << vector_to_print[i] << " | ";
  std::cout << std::endl;
}

//
//void
//Utils::remove_vector_of_keypoints_p2d_with_list_of_index(
//    const std::vector<int>& list_of_index,
//    FrameSharedPtr& frame)
//{
//  std::vector<cv::Point2f> cp;
//  for (int i=0; i<frame->keypoints_p2d.size(); i++)
//  {
//    bool take_element = true;
//    for (const int& index: list_of_index)
//    {
//      if (i == index)
//      {
//        take_element = false;
//        break;
//      }
//    }
//    if (take_element)
//    {
//      cp.push_back(frame->keypoints_p2d[i]);
//    }
//  }
//  frame->keypoints_p2d.clear();
//  frame->keypoints_p2d = cp;
//}

/*void
Utils::remove_vector_of_keypoints_with_list_of_index(
  const std::vector<int> &list_of_index,
  std::vector<cv::KeyPoint> &vector_to_remove_elements)
{
  for (int i = 0; i < list_of_index.size(); i++)
  {
    vector_to_remove_elements.erase(
      std::remove(vector_to_remove_elements.begin(),
                    vector_to_remove_elements.end(), list_of_index[i]),
      vector_to_remove_elements.end());
  }
}*/

} // eof MonocularVO
#include "maps.hpp"
#include <utility>


namespace MonocularVO
{


MonocularVO::Map::Map(
        MonocularVO::Params  params)
  : params_(std::move(params))
{

}

MonocularVO::FrameSharedPtr
MonocularVO::Map::get_curr_frame()
{
  return frames_.end()[-1];
}

MonocularVO::FrameSharedPtr
MonocularVO::Map::get_prev_frame()
{
  if (get_frame_count()<2)
  {
    throw std::runtime_error("Previous frame doesn't exist.");
  } else{
    return frames_.end()[-2];
  }
}



void
MonocularVO::Map::delete_past_images()
{
  for (
    auto it=frames_.begin();
    it!=frames_.end()-1;
    it++)
  {
    auto frame = *it;
    frame->image_gray.release();
    frame->image_gray_with_kpts.release();
    frame->img_colored.release();
    frame->is_img_deleted = true;
  }
}


void
MonocularVO::Map::push_frame(
  const MonocularVO::FrameSharedPtr &frame)
{
  frames_.push_back(frame);
}


int
MonocularVO::Map::get_frame_count()
{
  return frames_.size();
}

int
MonocularVO::Map::get_key_frame_count()
{
  int count_key_frame = 0;
  for (const auto& frame:frames_)
  {
    if (frame->is_key_frame)
      count_key_frame++;
  }
  return count_key_frame;
}

void
MonocularVO::Map::update_past_frames_optical_flow(
  const std::vector<uchar>& status)
{
  //getting rid of points for which the KLT tracking
  // failed or those who have gone outside the frame
  int x = frames_.front()->width;
  int y = frames_.front()->height;
  int indexCorrection = 0;
  for( int i=0; i<status.size(); i++)
  {
    cv::Point2f pt = get_curr_frame()->keypoints_pt.at(
    i- indexCorrection);
    if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0)||pt.x>x||pt.y>y)
    {
      for (auto it=frames_.begin()+get_last_key_frame_id_in_frames();
        it!=frames_.end(); it++)
      {
        FrameSharedPtr view = *it;
        view->keypoints_pt.erase(view->keypoints_pt.begin() + i - indexCorrection);
      }
      indexCorrection++;
    }
  }
  count_local_landmark_ = static_cast<int>(
    frames_.back()->keypoints_pt.size());
}


void
MonocularVO::Map::update_past_frames_epipolar(
  const cv::Mat& inliers_F,
  const cv::Mat& inliers_E)
{
  int indexCorrection = 0;
  for(size_t i=0; i<inliers_E.rows; i++)
  {
    if (inliers_F.at<bool>(i,0) == false or
        inliers_E.at<bool>(i,0) == false)
    {
      for (auto it=frames_.begin()+get_last_key_frame_id_in_frames();
           it!=frames_.end(); it++)
      {
        FrameSharedPtr view = *it;
        view->keypoints_pt.erase(view->keypoints_pt.begin() + i - indexCorrection);
      }
      indexCorrection++;
    }
  }
  count_local_landmark_ = static_cast<int>(
    frames_.back()->keypoints_pt.size());
}


void
MonocularVO::Map::print_frames_info()
{
  for (const FrameSharedPtr& frame:frames_)
    std::cout << "Frame id: " << frame->view_id << " "
    "| kpt count:" << frame->keypoints_pt.size() <<
    " | is key-frame: " << frame->is_key_frame <<
    " | key-point count: " << frame->keypoints.size() << std::endl;
std::cout << "Total frame count: " << get_frame_count() << std::endl;
std::cout << "Local landmark count: " << count_local_landmark_ << std::endl;
}


FrameSharedPtr
Map::get_last_key_frame()
{
  FrameSharedPtr last_key_frame;
  for (auto frame:frames_)
  {
    if (frame->is_key_frame)
      last_key_frame = frame;
  }
  return last_key_frame;
}

int
Map::get_last_key_frame_id_in_frames()
{
  std::vector<int> vector_ids_kyfrm;
  for(int i=0; i<get_frame_count(); i++)
  {
    if (frames_[i]->is_key_frame)
      vector_ids_kyfrm.push_back(i);
  }
  return vector_ids_kyfrm.end()[-1];
}

int
Map::get_prev_key_frame_id_in_frames()
{
  std::vector<int> vector_ids_kyfrm;
  for(int i=0; i<get_frame_count(); i++)
  {
    if (frames_[i]->is_key_frame)
      vector_ids_kyfrm.push_back(i);
  }
  return vector_ids_kyfrm.end()[-2];
}




Batch
Map::build_batch_1()
{
  Batch batch;
  FrameSharedPtr prev_key_frame = frames_[get_prev_key_frame_id_in_frames()];
  FrameSharedPtr curr_key_frame = get_curr_frame();
  int diff = curr_key_frame->view_id - prev_key_frame->view_id;

  FrameSharedPtr frame_middle = frames_[get_prev_key_frame_id_in_frames()+(diff/2)];
  // It will be use as a constraint
  frame_middle->set_key_frame();
  batch.push_back(prev_key_frame);
  batch.push_back(frame_middle);
  batch.push_back(curr_key_frame);

  for (const auto & frame : batch)
    std::cout << "View id : " << frame->view_id << " | Key-frame: " << frame->is_key_frame <<
    " | Feature extracted: "<< frame->is_feature_extracted <<std::endl;


  return batch;
}


Batch
Map::build_batch_2()
{
  Batch batch;
  FrameSharedPtr prev_key_frame = frames_[get_prev_key_frame_id_in_frames()];
  FrameSharedPtr curr_key_frame = get_curr_frame();
  batch.push_back(prev_key_frame);
  batch.push_back(curr_key_frame);

  for (const auto & frame : batch)
    std::cout << "View id : " << frame->view_id << std::endl;

  return batch;
}


} // eof MonocularVO

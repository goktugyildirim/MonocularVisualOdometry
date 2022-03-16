#include "frames.hpp"
#include <utility>


namespace MonocularVO
{

MonocularVO::Frames::Frames(
  MonocularVO::Params  params)
  : m_params(std::move(params)),
    m_id_last_frame(0)
{

}


void
Frames::push_frame(const FrameSharedPtr &frame)
{
  m_prev_frame = m_curr_frame;
  m_curr_frame = frame;
  m_frames[m_id_last_frame] = frame;
  m_id_last_frame++;
}


FrameSharedPtr
Frames::get_curr_frame()
{
  return m_curr_frame;
}

FrameSharedPtr Frames::get_prev_frame()
{
  return m_prev_frame;
}


void
Frames::set_curr_frame_is_ref_frame()
{
  m_curr_frame->set_ref_frame();
  m_ref_frame = m_curr_frame;
}


FrameSharedPtr
Frames::get_ref_frame()
{
  return m_ref_frame;
}


void
Frames::print_info()
{
  std::mutex mutex;
  mutex.lock();
  for (auto const& x : m_frames)
  {
    std::cout << "Frame id: " << x.first <<
      " | is ref frame: " << x.second->is_ref_frame <<
      " | is keyframe: " << x.second->is_keyframe <<
      " | count tracked keypoint: " << x.second->keypoints_p2d.size() <<
      std::endl;
  }
  mutex.unlock();
}

} // eof MonocularVO

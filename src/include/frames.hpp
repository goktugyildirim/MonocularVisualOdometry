#ifndef BUILD_SRC_INCLUDE_FRAMES_HPP_
#define BUILD_SRC_INCLUDE_FRAMES_HPP_
#include "params.hpp"
#include "frame.hpp"
#include "types.hpp"
#include <utils.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "opencv2/opencv.hpp"

#include "frames.hpp"

namespace MonocularVO
{


class Frames
{
public:
  using FrameSharedPtr = std::shared_ptr<Frame>;

  std::map<int, MonocularVO::FrameSharedPtr> m_frames;
  int m_id_last_frame;
  FrameSharedPtr m_curr_frame;
  FrameSharedPtr m_prev_frame;
  FrameSharedPtr m_ref_frame;
  MonocularVO::Params m_params;

  explicit Frames(MonocularVO::Params  params);

  void push_frame(const MonocularVO::FrameSharedPtr& frame);
  FrameSharedPtr get_curr_frame();
  FrameSharedPtr get_prev_frame();
  void set_curr_frame_is_ref_frame();
  FrameSharedPtr get_ref_frame();

  void print_info();


}; // eof Frames








}

#endif // BUILD_SRC_INCLUDE_FRAMES_HPP_

#ifndef BUILD_SRC_INCLUDE_INITIALIZER_HPP_
#define BUILD_SRC_INCLUDE_INITIALIZER_HPP_

#include <params.hpp>
#include <iostream>
#include <vector>
#include <future>
#include <mutex>
#include <thread>
#include <algorithm>
#include <memory>
#include "frame.hpp"
#include <types.hpp>
#include "vision.hpp"
#include "utils.hpp"
#include "local_handler.hpp"
#include "frames.hpp"

#include "concurrency/concurrentqueue.h"

#include <Eigen/Dense>

namespace MonocularVO
{

class LocalTrackingHandler {
 public:

  using FrameSharedPtr = std::shared_ptr<Frame>;
  using LockFreeQueue = moodycamel::ConcurrentQueue<std::shared_ptr<Frame>>;
  //using LockFreeQueueBatch = moodycamel::ConcurrentQueue<Batch>;
  using TypeCallbackTrack = std::function<void (const cv::Mat& img_concat)>;

  TypeCallbackTrack send_to_ros_interface;

  explicit LocalTrackingHandler(
      const MonocularVO::Params& params,
      TypeCallbackTrack&  callback_view_tracked);

  virtual ~LocalTrackingHandler();

  void start(std::shared_ptr<LockFreeQueue>& queue_view_to_initialization);

  void stop();

 private:
  MonocularVO::Params m_params;
  std::future<void> m_future_worker_local_tracking_handler;
  std::atomic_bool m_keep_tracking;

  void track_frames(std::shared_ptr<LockFreeQueue> &queue_view_to_tracking);
  void track_p2d_optical_flow(const int& window_size);
  // Important member variables:
/*  std::atomic_bool m_need_init;
  std::atomic_bool m_is_init_done;*/
  std::atomic_bool m_is_ref_frame_selected;
  std::vector<int> m_tracked_landmark_ids;
  Frames m_frames;



/*
  void try_send_batch_to_local_handler(Batch& batch);
  std::shared_ptr<LocalHandler> m_worker_local_handler;
  std::shared_ptr<LockFreeQueueBatch> m_queue_batch_to_local_handler;


*/





};

}

#endif  // BUILD_SRC_INCLUDE_INITIALIZER_HPP_

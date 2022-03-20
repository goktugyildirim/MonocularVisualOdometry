#ifndef BUILD_SRC_INCLUDE_LOCATRACKINGHANDLER_HPP_
#define BUILD_SRC_INCLUDE_LOCATRACKINGHANDLER_HPP_

#include "initializer.hpp"
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
#include <initializer.hpp>

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
  MonocularVO::Initializer m_initializer;

  struct TrackingEvaluation
  {
    bool is_tracking_ok = false;
    bool ready_for_trying_to_init = false;
    bool is_keyframe = false;
  };
  TrackingEvaluation m_tracking_evaluation;

  void make_reference_frame(FrameSharedPtr& curr_frame);
  void track_frames(std::shared_ptr<LockFreeQueue> &queue_view_to_tracking);
  void track_observations_optical_flow(const int& window_size, const double&repr_threshold);
  void show_tracking(const float& downs_ratio);
  TrackingEvaluation eval_tracking();

  std::atomic_bool m_is_init_done;
  std::atomic_bool m_is_ref_frame_selected;

  std::vector<int> m_tracked_p2d_ids;
  std::vector<int> m_tracked_p3d_ids;
  Frames m_frames;



/*
  void try_send_batch_to_local_handler(Batch& batch);
  std::shared_ptr<LocalHandler> m_worker_local_handler;
  std::shared_ptr<LockFreeQueueBatch> m_queue_batch_to_local_handler;


*/


};

}

#endif  // BUILD_SRC_INCLUDE_LOCATRACKINGHANDLER_HPP_

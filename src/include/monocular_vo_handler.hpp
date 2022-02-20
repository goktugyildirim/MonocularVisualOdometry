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
#include "view.hpp"
#include <types.hpp>
#include "vision.hpp"
#include "utils.hpp"
#include "local_handler.hpp"
#include "maps.hpp"

#include "concurrency/concurrentqueue.h"

#include <Eigen/Dense>

namespace MonocularVO
{

class MonocularVOHandler
{
 public:

  using FrameSharedPtr = std::shared_ptr<Frame>;
  using Batch = std::vector<FrameSharedPtr>;
  using LockFreeQueue = moodycamel::ConcurrentQueue<std::shared_ptr<Frame>>;
  using LockFreeQueueBatch = moodycamel::ConcurrentQueue<Batch>;
  using TypeCallbackTrack = std::function<void (const cv::Mat& img_concat)>;

  using MapSharedPtr = std::shared_ptr<Map>;

  TypeCallbackTrack provide_;

  explicit MonocularVOHandler(
      const MonocularVO::Params& params,
      TypeCallbackTrack&  callback_view_tracked);

  virtual ~MonocularVOHandler();

  void start(std::shared_ptr<LockFreeQueue>& queue_view_to_initialization);

  void stop();

 private:
  MonocularVO::Params params_;
  std::future<void> future_worker_initializer_;
  std::atomic_bool keep_initialization_;

  void do_monocular_vo(std::shared_ptr<LockFreeQueue> &queue_view_to_tracking);
  MapSharedPtr map_initial_;

  void try_send_batch_to_local_handler(Batch& batch);
  std::shared_ptr<LocalHandler> worker_local_handler_;
  std::shared_ptr<LockFreeQueueBatch> queue_batch_to_local_handler_;







};

}

#endif  // BUILD_SRC_INCLUDE_INITIALIZER_HPP_

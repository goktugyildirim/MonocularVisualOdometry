#ifndef BUILD_SRC_INCLUDE_LOCAL_HANDLER_HPP_
#define BUILD_SRC_INCLUDE_LOCAL_HANDLER_HPP_

#include "types.hpp"
#include "params.hpp"
#include <iostream>
#include "frame.hpp"
#include "frames.hpp"
#include <future>
#include <vision.hpp>
#include "optimization.hpp"
#include "concurrency/concurrentqueue.h"

namespace MonocularVO
{
class LocalHandler
{
public:

  using FrameSharedPtr = std::shared_ptr<Frame>;
  using LockFreeQueueBatch = moodycamel::ConcurrentQueue<std::vector<FrameSharedPtr>>;
  using Batch = std::vector<FrameSharedPtr>;

  explicit LocalHandler(const MonocularVO::Params& params);
  virtual ~LocalHandler();

  void start(std::shared_ptr<LockFreeQueueBatch>& queue_batch_to_local_handler);

  void stop();

private:
  std::mutex mutex_;
  MonocularVO::Params params_;
  std::future<void> future_worker_local_handler_;
  std::atomic_bool keep_local_handling_;
  void handle(std::shared_ptr<LockFreeQueueBatch> &queue_batch_to_local_handler);


  MonocularVO::LocalObservations build_local_observations(Batch& batch);




};





}


#endif  // BUILD_SRC_INCLUDE_LOCAL_HANDLER_HPP_

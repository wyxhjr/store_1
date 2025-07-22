#pragma once
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "IPCTensor.h"
#include "base/base.h"
#include "base/cu_utils.cuh"
#include "base/debug_utils.h"
#include "base/json.h"
#include "base/lock.h"
#include "base/pq.h"
#include "base/timer.h"
#include "cpu_embedding.h"
#include "parallel_pq.h"
#include "torch_utils.h"

// #define GRAD_ASYNC_V1_DEBUG
#define USE_SUB_GRAD_TENSOR
// #define XMH_DEBUG_KG
// #define USE_NEG_THREAD
#define USE_BOOST_SHUFFLE

namespace recstore {

class GraphEnv {
 public:
  static GraphEnv *instance_;

  static void Init(const std::string &json_str,
                   const std::vector<std::vector<int64_t>> &cached_range,
                   int64_t nr_graph_node) {
    if (instance_ == nullptr)
      instance_ = new GraphEnv(json_str, cached_range, nr_graph_node);
  }

  static GraphEnv *GetInstance() {
    CHECK(instance_ != nullptr);
    return instance_;
  }

 private:
  GraphEnv(const std::string &json_str,
           const std::vector<std::vector<int64_t>> &cached_range,
           int64_t nr_graph_node)
      : full_emb_("full_emb",
                  IPCTensorFactory::FindIPCTensorFromName("full_emb")) {
    cached_range_ = cached_range;
    auto json_config = json::parse(json_str);
    num_gpus_ = json_config.at("num_gpus");
    L_ = json_config.at("L");
    kForwardItersPerStep_ = json_config.at("kForwardItersPerStep");
    clr_ = json_config.at("clr");
    backgrad_init_ = json_config.at("backgrad_init");
    // full_emb_ = IPCTensorFactory::FindIPCTensorFromName("full_emb");

    nr_graph_node_ = nr_graph_node;
    LOG(WARNING) << folly::sformat("KGCacheController, config={}", json_str);
  }

 public:
  void RegTensorsPerProcess() {
    LOG(INFO) << "GraphEnv RegTensorsPerProcess";
    // IPCTensorFactory::ListIPCTensors();
    for (int rank = 0; rank < num_gpus_; ++rank) {
      nv::CudaDeviceRestorer _;
      CUDA_CHECK(cudaSetDevice(rank));
      input_keys_per_rank_.push_back(
          IPCTensorFactory::GetSlicedIPCTensorFromName(
              folly::sformat("input_keys_{}", rank)));
      input_keys_neg_per_rank_.push_back(
          IPCTensorFactory::GetSlicedIPCTensorFromName(
              folly::sformat("input_keys_neg_{}", rank)));

      // cache tensor
      cache_per_rank_.push_back(IPCTensorFactory::FindIPCTensorFromName(
          folly::sformat("embedding_cache_{}", rank)));

      // L buffer of input ids
      cached_id_circle_buffer_.push_back(
          std::vector<c10::intrusive_ptr<SlicedTensor>>());
      for (int j = 0; j < L_; ++j) {
        cached_id_circle_buffer_[rank].push_back(
            IPCTensorFactory::GetSlicedIPCTensorFromName(
                folly::sformat("cached_sampler_r{}_{}", rank, j)));
      }
      // step tensor
      step_tensor_per_rank_.push_back(IPCTensorFactory::FindIPCTensorFromName(
          folly::sformat("step_r{}", rank)));

      backward_grads_per_rank_.push_back(
          IPCTensorFactory::GetSlicedIPCTensorFromName(
              folly::sformat("backward_grads_{}", rank)));
      backward_grads_neg_per_rank_.push_back(
          IPCTensorFactory::GetSlicedIPCTensorFromName(
              folly::sformat("backward_grads_neg_{}", rank)));
      if (backgrad_init_ == "both") {
        backward_grads_gpu_.push_back(
            IPCTensorFactory::GetSlicedIPCTensorFromName(
                folly::sformat("backward_grads_{}_gpu", rank)));
        backward_grads_neg_gpu_.push_back(
            IPCTensorFactory::GetSlicedIPCTensorFromName(
                folly::sformat("backward_grads_neg_{}_gpu", rank)));
      }

      circle_buffer_end_per_rank_.push_back(
          IPCTensorFactory::FindIPCTensorFromName(
              folly::sformat("circle_buffer_end_r{}", rank)));
    }
  }

 public:
  //  config
  int num_gpus_;
  int L_;
  std::vector<std::vector<int64_t>> cached_range_;
  int kForwardItersPerStep_;
  float clr_;
  int64_t nr_graph_node_;

  // state tensor
  // torch::Tensor full_emb_;
  CPUEmbedding full_emb_;

  std::vector<torch::Tensor> cache_per_rank_;
  // runtime tensor
  std::vector<torch::Tensor> step_tensor_per_rank_;
  std::vector<torch::Tensor> circle_buffer_end_per_rank_;

  std::vector<c10::intrusive_ptr<SlicedTensor>> input_keys_per_rank_;
  std::vector<c10::intrusive_ptr<SlicedTensor>> input_keys_neg_per_rank_;
  std::vector<c10::intrusive_ptr<SlicedTensor>> backward_grads_per_rank_;
  std::vector<c10::intrusive_ptr<SlicedTensor>> backward_grads_neg_per_rank_;

  std::string backgrad_init_;
  // for backgrad_init_ = both
  std::vector<c10::intrusive_ptr<SlicedTensor>> backward_grads_gpu_;
  std::vector<c10::intrusive_ptr<SlicedTensor>> backward_grads_neg_gpu_;

  // different rank's L buffer
  std::vector<std::vector<c10::intrusive_ptr<SlicedTensor>>>
      cached_id_circle_buffer_;
};

class GradProcessingBase {
 public:
  GradProcessingBase(const std::string &json_str,
                     const std::vector<std::vector<int64_t>> &cached_range)
      : json_config_(json::parse(json_str)),
        full_emb_(GraphEnv::GetInstance()->full_emb_),
        cache_per_rank_(GraphEnv::GetInstance()->cache_per_rank_),
        step_tensor_per_rank_(GraphEnv::GetInstance()->step_tensor_per_rank_),
        circle_buffer_end_per_rank_(
            GraphEnv::GetInstance()->circle_buffer_end_per_rank_),
        input_keys_per_rank_(GraphEnv::GetInstance()->input_keys_per_rank_),
        input_keys_neg_per_rank_(
            GraphEnv::GetInstance()->input_keys_neg_per_rank_),
        backward_grads_per_rank_(
            GraphEnv::GetInstance()->backward_grads_per_rank_),
        backward_grads_neg_per_rank_(
            GraphEnv::GetInstance()->backward_grads_neg_per_rank_),
        backgrad_init_(GraphEnv::GetInstance()->backgrad_init_),
        // for both
        backward_grads_gpu_(GraphEnv::GetInstance()->backward_grads_gpu_),
        backward_grads_neg_gpu_(
            GraphEnv::GetInstance()->backward_grads_neg_gpu_),
        //
        cached_id_circle_buffer_(
            GraphEnv::GetInstance()->cached_id_circle_buffer_),
        processOneStepNegThread_(&GradProcessingBase::ProcessOneStepNegThread,
                                 this) {
#ifndef USE_NEG_THREAD
    processOneStepNegThread_.join();
#endif
    pthread_setname_np(pthread_self(), "ProcessBackward");
    cached_range_ = cached_range;
    num_gpus_ = json_config_.at("num_gpus");
    L_ = json_config_.at("L");
    kForwardItersPerStep_ = json_config_.at("kForwardItersPerStep");
    clr_ = json_config_.at("clr");
    update_cache_use_omp_ = json_config_.value("update_cache_use_omp", 1);
    update_pq_use_omp_ = json_config_.value("update_pq_use_omp", 2);
    nr_background_threads_ = json_config_.value("nr_background_threads", 8);

    if (backgrad_init_ == "cpu") {
      backgrad_init_enum_ = BackGradInitEnum::CPU;
    } else if (backgrad_init_ == "gpu") {
      backgrad_init_enum_ = BackGradInitEnum::GPU;
    } else if (backgrad_init_ == "both") {
      backgrad_init_enum_ = BackGradInitEnum::BOTH;
    } else {
      LOG(FATAL) << "invalide backgrad_init";
    }
    LOG(WARNING) << "Init GradProcessingBase done";
  }

  virtual void RegTensorsPerProcess() {
    GraphEnv::GetInstance()->RegTensorsPerProcess();
    CHECK(!isInitialized_);
    isInitialized_ = true;
  }

  static std::vector<torch::Tensor> split_keys_to_shards(
      const torch::Tensor keys,
      std::vector<std::vector<int64_t>> cached_range) {
    std::vector<torch::Tensor> in_each_rank_cache_mask;
    int num_gpus = cached_range.size();
    for (int shard_no = 0; shard_no < num_gpus; shard_no++) {
      int64_t start = cached_range[shard_no][0];
      int64_t end = cached_range[shard_no][1];

      torch::Tensor in_this_rank =
          keys.greater_equal(start).logical_and(keys.less(end));
      in_each_rank_cache_mask.push_back(in_this_rank);
    }
    return in_each_rank_cache_mask;
  }

  virtual void StopThreads() {
    stop_processOneStepNegThread_flag_.store(true);
    processOneStepNegThread_ping_.store(true);
  #ifdef USE_NEG_THREAD
    LOG(WARNING) << "before processOneStepNegThread_.join();";
    processOneStepNegThread_.join();
    LOG(WARNING) << "after processOneStepNegThread_.join();";
  #endif
  };

  virtual void StartThreads() {}

  virtual void ProcessOneStepNegThread() {
#ifdef USE_NEG_THREAD
    pthread_setname_np(pthread_self(), "ProcessBackward NegThread");
    LOG(WARNING) << "TID of ProcessBackward NegThread is "<< base::GetThreadId();
#else
    return;
#endif
    torch::AutoGradMode guard_false(false);
    if (kForwardItersPerStep_ == 1) {
      return;
    }
    while (!stop_processOneStepNegThread_flag_.load()) {
      while (processOneStepNegThread_ping_.load() == false)
        ;
      if (stop_processOneStepNegThread_flag_.load() == true) return;

      auto input_keys_neg_per_rank_tensors =
          SlicedTensor::BatchConvertToTensors(input_keys_neg_per_rank_);
      auto backward_grads_neg_per_rank_tensors =
          SlicedTensor::BatchConvertToTensors(backward_grads_neg_per_rank_);
      auto backward_grads_neg_per_rank_tensors_gpu =
          SlicedTensor::BatchConvertToTensors(backward_grads_neg_gpu_);

      if (backgrad_init_enum_ == BackGradInitEnum::BOTH) {
        UpdateCache(input_keys_neg_per_rank_tensors,
                    backward_grads_neg_per_rank_tensors_gpu);
      } else {
        UpdateCache(input_keys_neg_per_rank_tensors,
                    backward_grads_neg_per_rank_tensors);
      }

      ProcessBackward(input_keys_neg_per_rank_tensors,
                      backward_grads_neg_per_rank_tensors, now_step_.load());

      processOneStepNegThread_ping_ = false;
    }
  }

  virtual void ProcessOneStep(int64_t step_no) {
    torch::AutoGradMode guard_false(false);
    now_step_ = step_no;
#ifdef USE_NEG_THREAD
    if (kForwardItersPerStep_ > 1) {
      processOneStepNegThread_ping_.store(true);
    }
#endif

    auto input_keys_per_rank_tensors =
        SlicedTensor::BatchConvertToTensors(input_keys_per_rank_);
    auto backward_grads_per_rank_tensors =
        SlicedTensor::BatchConvertToTensors(backward_grads_per_rank_);
    auto backward_grads_per_rank_tensors_gpu =
        SlicedTensor::BatchConvertToTensors(backward_grads_gpu_);

    if (backgrad_init_enum_ == BackGradInitEnum::BOTH) {
      CHECK_EQ(backward_grads_per_rank_tensors.size(),
               backward_grads_per_rank_tensors_gpu.size());

      UpdateCache(input_keys_per_rank_tensors,
                  backward_grads_per_rank_tensors_gpu);
    } else {
      UpdateCache(input_keys_per_rank_tensors, backward_grads_per_rank_tensors);
    }

    ProcessBackward(input_keys_per_rank_tensors,
                    backward_grads_per_rank_tensors, step_no);

#ifdef USE_NEG_THREAD
    if (kForwardItersPerStep_ > 1) {
      while (processOneStepNegThread_ping_.load() == true)
        FB_LOG_EVERY_MS(ERROR, 5000) << "waiting for processOneStepNegThread_";
      return;
    }
#else
    if (kForwardItersPerStep_ > 1) {
      auto input_keys_neg_per_rank_tensors =
          SlicedTensor::BatchConvertToTensors(input_keys_neg_per_rank_);
      auto backward_grads_neg_per_rank_tensors =
          SlicedTensor::BatchConvertToTensors(backward_grads_neg_per_rank_);
      auto backward_grads_neg_per_rank_tensors_gpu =
          SlicedTensor::BatchConvertToTensors(backward_grads_neg_gpu_);

      if (backgrad_init_enum_ == BackGradInitEnum::BOTH) {
        UpdateCache(input_keys_neg_per_rank_tensors,
                    backward_grads_neg_per_rank_tensors_gpu);
      } else {
        UpdateCache(input_keys_neg_per_rank_tensors,
                    backward_grads_neg_per_rank_tensors);
      }

      ProcessBackward(input_keys_neg_per_rank_tensors,
                      backward_grads_neg_per_rank_tensors, step_no);
    }
#endif
  }

  virtual void ProcessBackward(const std::vector<torch::Tensor> &input_keys,
                               const std::vector<torch::Tensor> &input_grads,
                               int step_no) = 0;

  virtual void BlockToStepN(int step_no) = 0;

  /*-------- BOOST SHUFFLE VERSION ---------*/
  std::pair<std::vector<std::vector<torch::Tensor>>,
            std::vector<std::vector<torch::Tensor>>>
  Boost_ShuffleKeysAndGrads(const std::vector<torch::Tensor> &input_keys,
                            const std::vector<torch::Tensor> &input_grads) {
    int num_gpus_ = input_keys.size();
    // auto _start = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<torch::Tensor>> shuffled_keys_in_each_rank_cache;
    std::vector<std::vector<torch::Tensor>> shuffled_grads_in_each_rank_cache;
    shuffled_keys_in_each_rank_cache.resize(num_gpus_);
    shuffled_grads_in_each_rank_cache.resize(num_gpus_);
    for (int rank = 0; rank < num_gpus_; rank++) {
      shuffled_keys_in_each_rank_cache[rank].resize(num_gpus_);
      shuffled_grads_in_each_rank_cache[rank].resize(num_gpus_);
    }

#pragma omp parallel for num_threads(num_gpus_)
    for (int rank = 0; rank < num_gpus_; rank++) {
      for (int shard_no = 0; shard_no < num_gpus_; shard_no++) {
        torch::Tensor in_this_rank =
            input_keys[rank]
                .greater_equal(cached_range_[shard_no][0])
                .logical_and(input_keys[rank].less(cached_range_[shard_no][1]));
        shuffled_keys_in_each_rank_cache[shard_no][rank] =
            std::move(at::indexing::get_item(
                input_keys[rank], {at::indexing::TensorIndex(in_this_rank)}));
        shuffled_grads_in_each_rank_cache[shard_no][rank] =
            std::move(at::indexing::get_item(
                input_grads[rank], {at::indexing::TensorIndex(in_this_rank)}));
      }
    }
    // auto _end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> duration = _end - _start;
    // std::cout << "[BOOST] num_workers= " << num_gpus_
    //           << " Execution time: " << duration.count() << " seconds."
    //           << std::endl;

    return std::make_pair(shuffled_keys_in_each_rank_cache,
                          shuffled_grads_in_each_rank_cache);
  }

  //  cached_range:
  //        [ (start, end ),  # rank0
  //         (start, end),    # rank1
  //         (start, end),  ....
  //         (start, end),
  //         (start, end),    # rank7
  //        ]
  std::pair<std::vector<std::vector<torch::Tensor>>,
            std::vector<std::vector<torch::Tensor>>>
  ShuffleKeysAndGrads(const std::vector<torch::Tensor> &input_keys,
                      const std::vector<torch::Tensor> &input_grads) {
#ifdef USE_BOOST_SHUFFLE
    return Boost_ShuffleKeysAndGrads(input_keys, input_grads);
#endif
    // xmh::Timer timer_ShuffleKeysAndGrads("ProcessBack:Shuffle");
    std::vector<std::vector<torch::Tensor>> shuffled_keys_in_each_rank_cache;
    std::vector<std::vector<torch::Tensor>> shuffled_grads_in_each_rank_cache;
    shuffled_keys_in_each_rank_cache.resize(num_gpus_);
    shuffled_grads_in_each_rank_cache.resize(num_gpus_);

#pragma parallel for thread_num(num_gpus_)
    for (int rank = 0; rank < num_gpus_; rank++) {
      // CHECK(!input_keys[rank].is_cuda());
      // CHECK(input_grads[rank].is_cuda());

      auto in_each_rank_cache_mask =
          split_keys_to_shards(input_keys[rank], cached_range_);

      std::vector<torch::Tensor> sharded_keys_in_this_rank =
          TensorUtil::IndexVectors(input_keys[rank], in_each_rank_cache_mask);
      std::vector<torch::Tensor> sharded_grads_in_this_rank =
          TensorUtil::IndexVectorsDebug(input_grads[rank],
                                        in_each_rank_cache_mask);

// shuffle keys and grads
#pragma omp critical
      {
        for (int i = 0; i < num_gpus_; i++) {
          shuffled_keys_in_each_rank_cache[i].push_back(
              sharded_keys_in_this_rank[i]);
          shuffled_grads_in_each_rank_cache[i].push_back(
              sharded_grads_in_this_rank[i]);
        }
      }
    }
    // shuffle done
    return std::make_pair(shuffled_keys_in_each_rank_cache,
                          shuffled_grads_in_each_rank_cache);
  }

 private:
  virtual void UpdateCache(
      const std::vector<torch::Tensor> &input_keys_per_rank_tensors,
      const std::vector<torch::Tensor> &backward_grads_per_rank_tensors) {
    xmh::Timer timer_ShuffleKeysAndGrads("ProcessBack:CacheShuffle");
    auto [shuffled_keys_in_each_rank_cache, shuffled_grads_in_each_rank_cache] =
        ShuffleKeysAndGrads(input_keys_per_rank_tensors,
                            backward_grads_per_rank_tensors);
    timer_ShuffleKeysAndGrads.end();

    SyncUpdateCache(shuffled_keys_in_each_rank_cache,
                    shuffled_grads_in_each_rank_cache);
  }

  void SyncUpdateCache(const std::vector<std::vector<torch::Tensor>>
                           &shuffled_keys_in_each_rank_cache,
                       const std::vector<std::vector<torch::Tensor>>
                           &shuffled_grads_in_each_rank_cache) {
    xmh::Timer timer_SyncUpdateCache("ProcessBack:UpdateCache");
#pragma omp parallel for num_threads(num_gpus_) if (update_cache_use_omp_)
    for (int rank = 0; rank < num_gpus_; rank++) {
      for (int j = 0; j < num_gpus_; j++) {
        // update per rank cache
        // CUDA_CHECK(cudaSetDevice(rank));
        torch::Device device(torch::kCUDA, rank);
        auto in_rank_keys =
            shuffled_keys_in_each_rank_cache[rank][j].to(device, true) -
            cached_range_[rank][0];
        auto shuffle_grads_cuda =
            shuffled_grads_in_each_rank_cache[rank][j].to(device, true);
        cache_per_rank_[rank].index_add_(0, in_rank_keys,
                                         -clr_ * shuffle_grads_cuda);
      }
    }
    timer_SyncUpdateCache.end();
  }

 protected:
  // config
  nlohmann::json json_config_;
  int num_gpus_;
  int L_;
  std::vector<std::vector<int64_t>> cached_range_;
  int kForwardItersPerStep_;
  float clr_;

  // state tensor
  // torch::Tensor &full_emb_;
  CPUEmbedding &full_emb_;

  std::vector<torch::Tensor> &cache_per_rank_;
  bool isInitialized_ = false;

  enum BackGradInitEnum { CPU, GPU, BOTH };
  BackGradInitEnum backgrad_init_enum_;

  // runtime tensor
  std::vector<torch::Tensor> &step_tensor_per_rank_;
  std::vector<torch::Tensor> &circle_buffer_end_per_rank_;
  std::vector<c10::intrusive_ptr<SlicedTensor>> &input_keys_per_rank_;
  std::vector<c10::intrusive_ptr<SlicedTensor>> &input_keys_neg_per_rank_;

  std::vector<c10::intrusive_ptr<SlicedTensor>> &backward_grads_per_rank_;
  std::vector<c10::intrusive_ptr<SlicedTensor>> &backward_grads_neg_per_rank_;

  std::string backgrad_init_;
  // for backgrad_init_ = both
  std::vector<c10::intrusive_ptr<SlicedTensor>> &backward_grads_gpu_;
  std::vector<c10::intrusive_ptr<SlicedTensor>> &backward_grads_neg_gpu_;

  // different rank's L buffer
  std::vector<std::vector<c10::intrusive_ptr<SlicedTensor>>>
      &cached_id_circle_buffer_;

  std::atomic_int now_step_;
  std::thread processOneStepNegThread_;
  std::atomic_bool stop_processOneStepNegThread_flag_{false};
  std::atomic_bool processOneStepNegThread_ping_{false};
  int update_cache_use_omp_;
  int update_pq_use_omp_;
  int nr_background_threads_;
};

class GradSyncProcessing : public GradProcessingBase {
 public:
  GradSyncProcessing(const std::string &json_str,
                     const std::vector<std::vector<int64_t>> &cached_range)
      : GradProcessingBase(json_str, cached_range) {}

  void ProcessBackward(const std::vector<torch::Tensor> &input_keys,
                       const std::vector<torch::Tensor> &input_grads,
                       int step_no) override {
    xmh::Timer timer_ShuffleKeysAndGrads("ProcessBack:Shuffle");
    auto [shuffled_keys_in_each_rank, shuffled_grads_in_each_rank] =
        ShuffleKeysAndGrads(input_keys, input_grads);
    timer_ShuffleKeysAndGrads.end();

    SGDGradUpdate(shuffled_keys_in_each_rank, shuffled_grads_in_each_rank,
                  input_keys, input_grads);
  }

  void BlockToStepN(int step_no) {}

  void SGDGradUpdate(const std::vector<std::vector<torch::Tensor>>
                         &shuffled_keys_in_each_rank_cache,
                     const std::vector<std::vector<torch::Tensor>>
                         &shuffled_grads_in_each_rank_cache,
                     const std::vector<torch::Tensor> &input_keys,
                     const std::vector<torch::Tensor> &input_grads) {
    // SyncUpdateCache(shuffled_keys_in_each_rank_cache,
    //                 shuffled_grads_in_each_rank_cache);
#if 0
    for (int rank = 0; rank < num_gpus_; rank++) {
      for (int j = 0; j < num_gpus_; j++) {
        // LOG(WARNING) << rank << ": shuffled_keys_in_each_rank_cache[rank][j]
        // shape="
        //              << shuffled_keys_in_each_rank_cache[rank][j].sizes();
        LOG(WARNING) << rank << ": shuffled_keys_in_each_rank_cache[rank][j]"
                     << toString(shuffled_keys_in_each_rank_cache[rank][j],
                                 false);
        // LOG(WARNING) << rank << ": shuffled_grads_in_each_rank_cache[rank][j]
        // shape="
        //              << shuffled_grads_in_each_rank_cache[rank][j].sizes();
        LOG(WARNING) << rank << ": shuffled_grads_in_each_rank_cache[rank][j]"
                     << toString(shuffled_grads_in_each_rank_cache[rank][j],
                                 false);

      }
    }
#endif

    xmh::Timer timer_UpdateFull("ProcessBack:UpdateFull");
    // update full emb
    for (int rank = 0; rank < num_gpus_; rank++) {
      full_emb_.index_add_(0, input_keys[rank].cpu(),
                           (-clr_) * input_grads[rank].cpu());
    }
    timer_UpdateFull.end();
  }
};

}  // namespace recstore
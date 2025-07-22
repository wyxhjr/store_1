#pragma once
#include "folly/executors/CPUThreadPoolExecutor.h"
#include "grad_async_v1.h"
#include "grad_base.h"
#include "grad_memory_manager.h"
#include "parallel_pq_v2.h"
// #define XMH_DEBUG_KG

namespace recstore {

class GradAsyncProcessingV2 : public GradAsyncProcessing {
  static constexpr bool kUseBackThread_ = false;
  // static constexpr bool kUseBackThread_ = true;

 public:
  GradAsyncProcessingV2(const std::string &json_str,
                        const std::vector<std::vector<int64_t>> &cached_range)
      : GradAsyncProcessing(json_str, cached_range) {
    if (pq_) delete pq_;
    pq_ = new recstore::ParallelPqV2<AsyncGradElement>(
        kEmbNumber_, json_config_.value("kUseParallelClean", 1));
  }

  void StartThreads() override {
    CHECK(isInitialized_);
    dispatch_thread_stop_flag_ = false;
    grad_thread_stop_flag_ = false;

    if (kUseBackThread_) {
      for (int i = 0; i < nr_background_threads_; ++i) {
        backward_threads_.emplace_back(
            std::bind(&GradAsyncProcessingV2::GradWorkThread, this, i));
      }
    }
    dispatch_thread_ =
        std::thread(&GradAsyncProcessingV2::DispatchThread, this);

    detect_thread_ = std::thread(&GradAsyncProcessingV2::DetectThread, this);
  }

  void WhenNewSampleComes(c10::intrusive_ptr<SlicedTensor> input_keys, int rank,
                          int step_no) {
    // 来了一个新样本step号：把<里面的ID, step号>插堆
    auto *data = input_keys->GetSlicedTensor().data_ptr<int64_t>();
    for (int i = 0; i < input_keys->GetSlicedTensor().size(0); ++i) {
      int64_t id = data[i];
      auto *p = dict_[id];
      base::LockGuard element_lock_guard(*p);
      p->MarkReadInStepN(step_no);
      p->RecaculatePriority();
      pq_->PushOrUpdate(p);
    }
  }

  void DetectNewSamplesCome() {
    for (int rank = 0; rank < num_gpus_; rank++) {
      volatile int64_t *p_new_end =
          circle_buffer_end_per_rank_[rank].data_ptr<int64_t>();
      int64_t new_end = *p_new_end;
      volatile int64_t *p_old_end =
          circle_buffer_end_cppseen_[rank].data_ptr<int64_t>();
      int64_t old_end = *p_old_end;

      if (new_end != old_end) {
        FB_LOG_EVERY_MS(WARNING, 1000) << folly::sformat(
            "Detect new sample comes, old_end{}, new_end{}", old_end, new_end);

        // add [circle_buffer_old_end, new_end)
        if (new_end < old_end) new_end += L_;
        for (int i = old_end; i < new_end; ++i) {
          int pointer = (i % L_);
          int step = step_tensor_per_rank_[rank][pointer].item<int64_t>();
          WhenNewSampleComes(cached_id_circle_buffer_[rank][pointer], rank,
                             step);

          // update the seen sample step of detect threads
          CHECK_LT(sample_step_cpp_seen_[rank], step);
          sample_step_cpp_seen_[rank] = step;
        }
        new_end = new_end % L_;
        *p_old_end = new_end;
      }
    }
  }

  void ChunkCleanHelper(AsyncGradElement *p) {
    CHECK_EQ(p->magic_, 0xdeadbeef);
    int64_t id = p->GetID();

    p->Lock();
    auto [no_used, vec_grads] = p->DrainWrites();
    p->Unlock();

#ifdef USE_SUB_GRAD_TENSOR
    std::vector<torch::Tensor> grads;
    for (auto &each : vec_grads) {
      grads.push_back(each.to_tensor());
    }
#else
    auto &grads = vec_grads;
#endif

    static int round_robin = 0;
    for (int i = 0; i < grads.size(); i++) {
      if (kUseBackThread_) {
        LOG(FATAL) << "now don't use backthread";
        while (!backthread_work_queues_[round_robin]->write(
            std::make_pair(id, grads[i]))) {
          continue;
        }
        round_robin = (round_robin + 1) % nr_background_threads_;
      } else {
        auto grad = grads[i].cpu().unsqueeze_(0);
#ifdef XMH_DEBUG_KG
        LOG(INFO) << "+Grad: "
                  << "| " << p->ToString() << "|" << toString(full_emb_[id])
                  << " -> " << toString(full_emb_[id] - clr_ * grad.squeeze(0));
#endif
        // NOTE(xieminhui): the following code is not thread-safe, may cause
        // inconsistent result compared with std embedding
        {
          // full_emb_.index_add_(0, torch::full({1}, id), -clr_ * grad);
          full_emb_.safe_index_add_(id, -clr_ * grad);
        }
      }
    }
    p->Lock();
    p->RecaculatePriority();
    pq_->PushOrUpdate(p);
    p->Unlock();
  }

  void DetectThread() {
    pthread_setname_np(pthread_self(), "DetectThread");

    torch::AutoGradMode guard_false(false);
    CHECK(isInitialized_);
    while (!detect_thread_stop_flag_.load()) {
      base::LockGuard _(large_lock_);
      DetectNewSamplesCome();
    }
  }

  void DispatchThread() {
    pthread_setname_np(pthread_self(), "DispatchThread");

    torch::AutoGradMode guard_false(false);
    CHECK(isInitialized_);
    while (!dispatch_thread_stop_flag_.load()) {
      base::LockGuard _(large_lock_);

      // DetectNewSamplesCome();

      // 后台线程，不断取堆头，dispatch给worker
      if (pq_->empty()) {
        continue;
      }

      // Chunk clean
      pq_->ChunkClean(std::bind(&GradAsyncProcessingV2::ChunkCleanHelper, this,
                                std::placeholders::_1));
    }
  }

  void GradWorkThread(int thread_id) {
    std::string thread_name = "GradWorkThread" + std::to_string(thread_id);
    pthread_setname_np(pthread_self(), thread_name.c_str());

    torch::AutoGradMode guard_false(false);
    auto *queue = backthread_work_queues_[thread_id].get();
    CHECK(queue != nullptr);
    while (!grad_thread_stop_flag_.load()) {
      std::pair<int64_t, torch::Tensor> p;
      while (!queue->read(p)) {
        // spin until we get a value
        if (grad_thread_stop_flag_.load()) {
          return;
        }
        continue;
      }
      int64_t id = p.first;
      torch::Tensor grad = p.second;
      grad = grad.cpu().unsqueeze_(0);
      full_emb_.index_add_(0, torch::full({1}, id), -clr_ * grad);
    }
  }

  // 等到可以让step_no开始训练
  void BlockToStepN(int step_no) override {
    // wait until the top element of heap > step_no
    while (true) {
      base::LockGuard _(large_lock_);

      if (pq_->empty()) {
        // LOG(WARNING) << "pq is empty";
        break;
      }

      int priority = pq_->MinPriority();
      if (priority > step_no) {
#ifdef XMH_DEBUG_KG
        LOG(INFO) << folly::sformat("top(pq)'s priority={} > step_no{}.",
                                    priority, step_no)
                  << pq_->ToString();
#endif
        break;
      }
      FB_LOG_EVERY_MS(WARNING, 1000)
          << "Sleep in <BlockToStepN>, step_no=" << step_no
          << ", pq.top=" << priority;
    }
  }

  void UpsertPqThreadMethod2ThreadPool(
      const std::vector<torch::Tensor> &input_keys,
      const std::vector<torch::Tensor> &input_grads, int step_no,
      int thread_id) {
    for (int rank = 0; rank < input_keys.size(); ++rank) {
      auto *data = input_keys[rank].data_ptr<int64_t>();
      CHECK(input_keys[rank].is_cpu());
      CHECK_EQ(input_grads[rank].dim(), 2);

      auto [thread_start, thread_end] =
          base::WorkParititon::MultiThreadWorkPartititon(
              thread_id, kUpdatePqWorkerNum_, input_keys[rank].size(0));

      for (int i = thread_start; i < thread_end; ++i) {
        int64_t id = data[i];

#ifdef USE_SUB_GRAD_TENSOR
        recstore::SubGradTensor grad_tensor(input_grads[rank], i);
#else
#error "not use SubGradTensor "
        torch::Tensor grad_tensor = input_grads[rank][i];
#endif

        auto *p = dict_[id];
        // NOTE: 改了优先级
        base::LockGuard element_lock_guard(*p);
        p->RemoveReadStep(step_no);
        p->MarkWriteInStepN(step_no, grad_tensor);
        p->RecaculatePriority();
#ifdef XMH_DEBUG_KG
        LOG(INFO) << folly::sformat("Push pq_ | id={}, step_no={}, grad={}", id,
                                    step_no, toString(grad_tensor));
#endif
        pq_->PushOrUpdate(p);
      }
    }
  }

  void UpsertPq(const std::vector<torch::Tensor> &input_keys,
                const std::vector<torch::Tensor> &input_grads, int step_no) {
    xmh::Timer timer_ProcessBackwardAsync("ProcessBack:UpsertPq");
    if (update_pq_use_omp_ == 0 || update_pq_use_omp_ == 1) {
#pragma omp parallel for num_threads(num_gpus_) if (update_pq_use_omp_)
      for (int rank = 0; rank < input_keys.size(); ++rank) {
        auto *data = input_keys[rank].data_ptr<int64_t>();
        CHECK(input_keys[rank].is_cpu());
        CHECK_EQ(input_grads[rank].dim(), 2);

        for (int i = 0; i < input_keys[rank].size(0); ++i) {
          int64_t id = data[i];

#ifdef USE_SUB_GRAD_TENSOR
          recstore::SubGradTensor grad_tensor(input_grads[rank], i);
#else
#error "not use SubGradTensor "
          torch::Tensor grad_tensor = input_grads[rank][i];
#endif

          auto *p = dict_[id];
          // NOTE: 改了优先级
          base::LockGuard element_lock_guard(*p);
          p->RemoveReadStep(step_no);
          p->MarkWriteInStepN(step_no, grad_tensor);
          p->RecaculatePriority();
#ifdef XMH_DEBUG_KG
          LOG(INFO) << folly::sformat("Push pq_ | id={}, step_no={}, grad={}",
                                      id, step_no,
                                      toString(grad_tensor, false));
#endif
          pq_->PushOrUpdate(p);
        }
      }
    } else if (update_pq_use_omp_ == 2) {
      for (int i = 0; i < kUpdatePqWorkerNum_; i++) {
        update_pq_thread_pool_->add(
            std::bind(&GradAsyncProcessingV2::UpsertPqThreadMethod2ThreadPool,
                      this, input_keys, input_grads, step_no, i));
      }
      update_pq_thread_pool_->join();
    } else {
      LOG(FATAL) << "invalid update_pq_use_omp_";
    }
    timer_ProcessBackwardAsync.end();
  }

  void ProcessBackward(const std::vector<torch::Tensor> &input_keys,
                       const std::vector<torch::Tensor> &input_grads,
                       int step_no) override {
    base::LockGuard _(large_lock_);
    // record the update
    // 把 <ID>查一下堆，拿一下step号
    // 如果不在堆，就插堆<ID, +无穷>，把grad指针填进去
    // 如果在堆，建立映射，把grad指针填进去

    for (int rank = 0; rank < num_gpus_; rank++) {
      while (sample_step_cpp_seen_[rank].load() < step_no)
        FB_LOG_EVERY_MS(ERROR, 5000)
            << "Stalled in ProcessBackward: "
            << folly::sformat(
                   "rank={}, step_no={}, "
                   "sample_step_cpp_seen_[rank]={}",
                   rank, step_no, sample_step_cpp_seen_[rank].load());
    }

    UpsertPq(input_keys, input_grads, step_no);
  }

 private:
};

}  // namespace recstore
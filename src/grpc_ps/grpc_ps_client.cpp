#include "grpc_ps_client.h"

#include <fmt/core.h>
#include <grpcpp/grpcpp.h>

#include <cstdint>
#include <future>
#include <string>
#include <vector>

#include "base/array.h"
#include "base/flatc.h"
#include "base/log.h"
#include "base/timer.h"
#include "base_ps/parameters.h"
#include "folly/executors/CPUThreadPoolExecutor.h"
#include "ps.grpc.pb.h"
#include "ps.pb.h"

using grpc::Channel;
using grpc::ClientAsyncResponseReader;
using grpc::ClientContext;
using grpc::Status;
using recstoreps::CommandRequest;
using recstoreps::CommandResponse;
using recstoreps::GetParameterRequest;
using recstoreps::GetParameterResponse;
using recstoreps::PSCommand;
using recstoreps::PutParameterRequest;
using recstoreps::PutParameterResponse;

DEFINE_int32(get_parameter_threads, 4, "get clients per shard");
DEFINE_bool(parameter_client_random_init, false, "");

// 新的构造函数，接收 json 配置参数
/*  
可用下面的代码来读取配置文件
std::ifstream config_file(FLAGS_config_path);
  nlohmann::json ex;
  config_file >> ex;
  json client_config = ex["client"];
  
*/
GRPCParameterClient::GRPCParameterClient(json config) : recstore::BasePSClient(config) {
  // 从json配置中提取参数
  host_       = config.value("host", "localhost");
  port_       = config.value("port", 15000);
  shard_      = config.value("shard", 0);
  nr_clients_ = FLAGS_get_parameter_threads;
  Initialize();
  channel_ = grpc::CreateChannel(fmt::format("{}:{}", host_, port_), grpc::InsecureChannelCredentials());
  for (int i = 0; i < nr_clients_; i++) {
    stubs_.push_back(nullptr);
    stubs_[i] = recstoreps::ParameterService::NewStub(channel_);
    LOG(INFO) << "Init PS Client Shard " << i;
  }
}

// 保留原有的构造函数以保持向后兼容
GRPCParameterClient::GRPCParameterClient(const std::string& host, int port, int shard)
    : recstore::BasePSClient(json{{"host", host}, {"port", port}, {"shard", shard}}),
      host_(host),
      port_(port),
      shard_(shard),
      nr_clients_(FLAGS_get_parameter_threads) {
  Initialize();
  channel_ = grpc::CreateChannel(fmt::format("{}:{}", host, port),
                                 grpc::InsecureChannelCredentials());
  for (int i = 0; i < nr_clients_; i++) {
    stubs_.push_back(nullptr);
    stubs_[i] = recstoreps::ParameterService::NewStub(channel_);
    LOG(INFO) << "Init PS Client Shard " << i;
  }
}

bool GRPCParameterClient::GetParameter(const ConstArray<uint64_t> &keys,
                                       float *values) {
  if (FLAGS_parameter_client_random_init) {
    CHECK(0) << "todo implement";
    return true;
  }

  get_param_key_sizes_.clear();
  get_param_status_.clear();
  get_param_requests_.clear();
  get_param_responses_.clear();
  get_param_resonse_readers_.clear();

  int request_num =
      (keys.Size() + MAX_PARAMETER_BATCH - 1) / MAX_PARAMETER_BATCH;
  get_param_status_.resize(request_num);
  get_param_requests_.resize(request_num);
  get_param_responses_.resize(request_num);

  for (int start = 0, index = 0; start < keys.Size();
       start += MAX_PARAMETER_BATCH, ++index) {
    int key_size = std::min((int)(keys.Size() - start), MAX_PARAMETER_BATCH);
    get_param_key_sizes_.emplace_back(key_size);
    auto &status = get_param_status_[index];
    auto &request = get_param_requests_[index];
    auto &response = get_param_responses_[index];
    request.set_keys(reinterpret_cast<const char *>(&keys[start]),
                     sizeof(uint64_t) * key_size);
    // rpc
    grpc::ClientContext context;
    std::unique_ptr<ClientAsyncResponseReader<GetParameterResponse>> rpc =
        stubs_[0]->AsyncGetParameter(&context, request, &cq);
    // GetParameter(&context, request, &response);
    rpc->Finish(&response, &status, reinterpret_cast<void *>(index));
  }
  int get = 0;
  while (get != request_num) {
    void *got_tag;
    bool ok = false;
    cq.Next(&got_tag, &ok);
    if (!ok) {
      LOG(ERROR) << "error";
    }
    get++;
  }
  size_t get_embedding_acc = 0;
  int old_dimension = -1;

  for (int i = 0; i < get_param_responses_.size(); ++i) {
    auto &response = get_param_responses_[i];
    int key_size = get_param_key_sizes_[i];
    auto parameters = reinterpret_cast<const ParameterCompressReader *>(
        response.parameter_value().data());

    if (parameters->size != key_size) {
      LOG(ERROR) << "GetParameter error: " << parameters->size << " vs "
                 << key_size;
      return false;
    }

    for (int index = 0; index < parameters->item_size(); ++index) {
      auto item = parameters->item(index);
      if (item->dim != 0) {
        if (old_dimension == -1) old_dimension = item->dim;
        CHECK_EQ(item->dim, old_dimension);
        std::copy_n(item->embedding, item->dim,
                    values + item->dim * get_embedding_acc);
        // values->emplace_back(
        //     std::vector<float>(item->embedding, item->embedding +
        //     item->dim));
      } else {
        FB_LOG_EVERY_MS(ERROR, 2000)
            << "error; not find key " << keys[get_embedding_acc] << " in ps";
      }
      get_embedding_acc++;
    }
  }
  return true;
}

bool GRPCParameterClient::GetParameter(
    const ConstArray<uint64_t> &keys, std::vector<std::vector<float>> *values) {
  if (FLAGS_parameter_client_random_init) {
    values->clear();
    values->reserve(keys.Size());
    for (size_t i = 0; i < keys.Size(); i++)
      values->emplace_back(std::vector<float>(128, 0.1));

    return true;
  }

  values->clear();
  get_param_key_sizes_.clear();
  get_param_status_.clear();
  get_param_requests_.clear();
  get_param_responses_.clear();
  get_param_resonse_readers_.clear();

  values->reserve(keys.Size());

  int request_num =
      (keys.Size() + MAX_PARAMETER_BATCH - 1) / MAX_PARAMETER_BATCH;

  get_param_status_.resize(request_num);
  get_param_requests_.resize(request_num);
  get_param_responses_.resize(request_num);

  for (int start = 0, index = 0; start < keys.Size();
       start += MAX_PARAMETER_BATCH, ++index) {
    int key_size = std::min((int)(keys.Size() - start), MAX_PARAMETER_BATCH);
    get_param_key_sizes_.emplace_back(key_size);
    auto &status = get_param_status_[index];
    auto &request = get_param_requests_[index];
    auto &response = get_param_responses_[index];
    request.set_keys(reinterpret_cast<const char *>(&keys[start]),
                     sizeof(uint64_t) * key_size);
    // rpc
    grpc::ClientContext context;
    get_param_resonse_readers_.emplace_back(
        stubs_[0]->AsyncGetParameter(&context, request, &cq));
    auto &rpc = get_param_resonse_readers_.back();
    // GetParameter(&context, request, &response);
    rpc->Finish(&response, &status, reinterpret_cast<void *>(index));
  }

  int get = 0;
  while (get != request_num) {
    void *got_tag;
    bool ok = false;
    cq.Next(&got_tag, &ok);
    if (unlikely(!ok)) {
      LOG(ERROR) << "error";
    }
    get++;
  }

  for (int i = 0; i < get_param_responses_.size(); ++i) {
    auto &response = get_param_responses_[i];
    int key_size = get_param_key_sizes_[i];
    auto parameters = reinterpret_cast<const ParameterCompressReader *>(
        response.parameter_value().data());

    if (unlikely(parameters->size != key_size)) {
      LOG(ERROR) << "GetParameter error: " << parameters->size << " vs "
                 << key_size;
      return false;
    }

    for (int index = 0; index < parameters->item_size(); ++index) {
      auto item = parameters->item(index);
      if (item->dim != 0) {
        values->emplace_back(
            std::vector<float>(item->embedding, item->embedding + item->dim));
      } else {
        values->emplace_back(std::vector<float>(0));
      }
    }
  }
  return true;
}

bool GRPCParameterClient::ClearPS() {
  CommandRequest request;
  CommandResponse response;
  request.set_command(PSCommand::CLEAR_PS);
  grpc::ClientContext context;
  grpc::Status status = stubs_[0]->Command(&context, request, &response);
  return status.ok();
}

bool GRPCParameterClient::LoadFakeData(int64_t data) {
  CommandRequest request;
  CommandResponse response;
  request.set_command(PSCommand::LOAD_FAKE_DATA);
  request.add_arg1(&data, sizeof(int64_t));
  grpc::ClientContext context;
  grpc::Status status = stubs_[0]->Command(&context, request, &response);
  return status.ok();
}

bool GRPCParameterClient::LoadCkpt(
    const std::vector<std::string> &model_config_path,
    const std::vector<std::string> &emb_file_path) {
  CommandRequest request;
  CommandResponse response;
  request.set_command(PSCommand::RELOAD_PS);

  for (auto &each : model_config_path) {
    request.add_arg1(each);
  }
  for (auto &each : emb_file_path) {
    request.add_arg2(each);
  }
  grpc::ClientContext context;
  grpc::Status status = stubs_[0]->Command(&context, request, &response);
  return status.ok();
}

bool GRPCParameterClient::PutParameter(
    const std::vector<uint64_t> &keys,
    const std::vector<std::vector<float>> &values) {
  for (int start = 0, index = 0; start < keys.size();
       start += MAX_PARAMETER_BATCH, ++index) {
    int key_size = std::min((int)(keys.size() - start), MAX_PARAMETER_BATCH);
    auto ret = std::make_shared<std::promise<bool>>();
    PutParameterRequest request;
    PutParameterResponse response;
    ParameterCompressor compressor;
    std::vector<std::string> blocks;
    for (int i = start; i < start + key_size; i++) {
      auto each_key = keys[i];
      auto &embedding = values[i];
      ParameterPack parameter_pack;
      parameter_pack.key = each_key;
      parameter_pack.dim = embedding.size();
      parameter_pack.emb_data = embedding.data();
      compressor.AddItem(parameter_pack, &blocks);
    }
    compressor.ToBlock(&blocks);
    CHECK_EQ(blocks.size(), 1);
    request.mutable_parameter_value()->swap(blocks[0]);
    grpc::ClientContext context;
    grpc::Status status = stubs_[0]->PutParameter(&context, request, &response);
    if (status.ok()) {
      ret->set_value(true);
    } else {
      std::cout << status.error_code() << ": " << status.error_message()
                << std::endl;
      ret->set_value(false);
    }
  }
  return true;
}

// 实现 BasePSClient 的纯虚函数
int GRPCParameterClient::GetParameter(const base::ConstArray<uint64_t>& keys, float* values) {
  bool success = GetParameter(keys, values);
  return success ? 0 : -1;
}

int GRPCParameterClient::AsyncGetParameter(const base::ConstArray<uint64_t>& keys, float* values) {
  
  return GetParameter(keys, values);
}

int GRPCParameterClient::PutParameter(const base::ConstArray<uint64_t>& keys,
                                      const std::vector<std::vector<float>>& values) {
  std::vector<uint64_t> key_vec(keys.Data(), keys.Data() + keys.Size());
  bool success = PutParameter(key_vec, values);
  return success ? 0 : -1;
}

//这个的作用是什么，需要如何修改
void GRPCParameterClient::Command(recstore::PSCommand command) {
  switch (command) {
  case recstore::PSCommand::CLEAR_PS:
    ClearPS();
    break;
  case recstore::PSCommand::RELOAD_PS:
    //  这里是要用GRPCParameterClient::LoadCkpt吗？
    LOG(WARNING) << "RELOAD_PS command requires additional parameters";
    break;
  case recstore::PSCommand::LOAD_FAKE_DATA:
    {
      int64_t fake_data = 1000;
      LoadFakeData(fake_data);
    }
    break;
  default:
    LOG(ERROR) << "Unknown PS command: " << static_cast<int>(command);
    break;
  }
}

// 注册 GRPCParameterClient 到工厂
FACTORY_REGISTER(recstore::BasePSClient, grpc, GRPCParameterClient, json);
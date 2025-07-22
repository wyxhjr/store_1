// #include <torch/custom_class.h>
// #include <torch/extension.h>
// #include <torch/torch.h>

// #include "grpc_ps_client.h"

// class GRPCParameterClientTorch : public torch::CustomClassHolder,
//                                  public GRPCParameterClient {
//  public:
//   explicit GRPCParameterClientTorch(const std::string &host, int64_t port,
//                                     int64_t shard)
//       : GRPCParameterClient(host, port, shard) {}

//   torch::Tensor GetParameter(torch::Tensor &keys, int64_t emb_dim,
//                              bool perf = true) {
//     const uint64_t *key_ptr = static_cast<const uint64_t *>(keys.data_ptr());
//     torch::Tensor result = torch::empty({keys.size(0), emb_dim});
//     float *value_ptr = static_cast<float *>(result.data_ptr());
//     ConstArray<uint64_t> keys_array(key_ptr, keys.size(0));
//     GRPCParameterClient::GetParameter(keys_array, value_ptr, perf);
//     return result;
//   }

//   bool PutParameter(torch::Tensor &keys, torch::Tensor &values) {
//     auto keys_accessor = keys.accessor<int64_t, 1>();
//     auto values_accessor = values.accessor<float, 2>();
//     for (int start = 0, index = 0; start < keys.size(0);
//          start += MAX_PARAMETER_BATCH, ++index) {
//       int key_size = std::min((int)(keys.size(0) - start), MAX_PARAMETER_BATCH);

//       PutParameterRequest request;
//       PutParameterResponse response;
//       ParameterCompressor compressor;
//       std::vector<std::string> blocks;
//       for (int i = start; i < start + key_size; i++) {
//         ParameterPack parameter_pack;
//         parameter_pack.key = keys_accessor[i];
//         parameter_pack.dim = values_accessor[i].size(0);
//         parameter_pack.emb_data =
//             static_cast<float *>(values_accessor[i].data());
//         compressor.AddItem(parameter_pack, &blocks);
//       }
//       compressor.ToBlock(&blocks);
//       CHECK_EQ(blocks.size(), 1);
//       request.mutable_parameter_value()->swap(blocks[0]);
//       grpc::ClientContext context;
//       grpc::Status status =
//           stubs_[0]->PutParameter(&context, request, &response);
//       if (!status.ok()) {
//         std::cout << status.error_code() << ": " << status.error_message()
//                   << std::endl;
//       }
//     }
//     return true;
//   }

//   bool LoadFakeData(int64_t data_size) {
//     return GRPCParameterClient::LoadFakeData(data_size);
//   }

//   bool ClearPS() { return GRPCParameterClient::ClearPS(); }
// };

// TORCH_LIBRARY(grpc_ps_client_python, m) {
//   m.class_<GRPCParameterClientTorch>("GRPCParameterClientTorch")
//       .def(torch::init<const std::string &, int64_t, int64_t>())
//       .def("GetParameter", &GRPCParameterClientTorch::GetParameter)
//       .def("PutParameter", &GRPCParameterClientTorch::PutParameter)
//       .def("LoadFakeData", &GRPCParameterClientTorch::LoadFakeData)
//       .def("ClearPS", &GRPCParameterClientTorch::ClearPS);
// }
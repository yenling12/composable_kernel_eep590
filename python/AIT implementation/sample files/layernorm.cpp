
size_t GLOBAL_WORKSPACE_SIZE = 0;

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <random>
#include <rocrand/rocrand.h>
#include "include/ck/utility/print.hpp"
#include "library/include/ck/library/utility/device_memory.hpp"
#include "library/include/ck/library/utility/host_tensor.hpp"
#include "library/include/ck/library/utility/host_tensor_generator.hpp"
#include "include/ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "include/ck/utility/reduction_operator.hpp"

#include "include/ck/tensor_operation/gpu/device/device_layernorm_impl.hpp"





using layernorm_rank2_256_1_256_1_8_1_8_1_8_1_8_8 = ck::tensor_operation::device::DeviceLayernormImpl<
    ck::half_t,
    ck::half_t,
    ck::half_t,
    float,
    ck::half_t,
    ck::tensor_operation::element_wise::PassThrough,
    2,
    1,
    
    256, // block_size
    1, // m_cluster_size
    256, // k_cluster_size
    1, // m_slice_size
    8, // k_slice_size
    1, // in_src_dim
    8, // in_src_size
    1, // gamma_src_dim
    8, // gamma_src_size
    1, // beta_src_dim
    8, // beta_src_size
    8 // out_dst_size

    >;
using DeviceInstance = layernorm_rank2_256_1_256_1_8_1_8_1_8_1_8_8;

void layernorm_0(void* input,
                   void* gamma,
                   void* beta,
                   void* output,

                   int64_t* in_0,

                   int64_t* in_1,

                   hipStream_t stream)
{

    int M = *in_0;
    int N = *in_1;
    

    std::vector<ck::index_t> i_inStrides;

    i_inStrides.push_back(N);
    i_inStrides.push_back(1);


    auto device_instance = DeviceInstance{};
    auto argument_ptr = device_instance.MakeArgumentPointer(
        {M, N},
        i_inStrides,
        std::vector<ck::index_t>{0, 1},
        std::vector<ck::index_t>{0, 1},
        i_inStrides,
        {1},
        1e-05,
        static_cast<ck::half_t *>(input),
        static_cast<ck::half_t *>(gamma),
        static_cast<ck::half_t *>(beta),
        static_cast<ck::half_t *>(output),
        ck::tensor_operation::element_wise::PassThrough{}
    );

    if(!device_instance.IsSupportedArgument(argument_ptr.get()))
    {
        throw std::runtime_error(
            "wrong! device_layernorm with the specified compilation parameters does "
            "not support this Softmax problem");
    };
    std::string instance_name = device_instance.GetTypeString();
    auto invoker_ptr = device_instance.MakeInvokerPointer();
    invoker_ptr->Run(argument_ptr.get(), StreamConfig{stream, false});
    return;
}
    



struct ProfilerMemoryPool {
  ProfilerMemoryPool() {
    std::random_device rd;
    gen = std::mt19937(rd());
    uniform_dist = std::uniform_int_distribution<int64_t>(1, 48964896);
    offsets.reserve(512);
    strides.reserve(512);
    copies.reserve(512);
    ptrs.reserve(512);
  }
  ~ProfilerMemoryPool() {
    for(int i = 0; i < ptrs.size(); i++){
      hipFree(ptrs[i]);
    }
  }

  template <typename DType>
  DType* AllocateGaussianTensor(int64_t size) {
    size_t length = size * sizeof(DType);
    DType *d_x;
    hipMalloc(&d_x, length);

    float mean = 0.0f;
    float stddev = 1.0f;
    uint64_t seed = uniform_dist(gen);
    rocrand_set_seed(generator, seed);
    rocrand_generate_normal(generator, reinterpret_cast<float*>(d_x), size, mean, stddev);
    return d_x;
  }

  ck::half_t* AllocateHalfGaussianTensor(int64_t size) {
    return reinterpret_cast<ck::half_t*>(
        AllocateGaussianTensor<ck::half_t>(size));
  }

  int AllocateHalfTensor(int64_t size, int64_t copy) {
    offsets.push_back(0);
    strides.push_back(size);
    copies.push_back(copy);
    auto ptr = AllocateHalfGaussianTensor(size * copy);
    ptrs.push_back(reinterpret_cast<void*>(ptr));
    return ptrs.size() - 1;
  }

  ck::half_t* RequestHalfTensorByIdx(int idx) {
    auto copy = copies.at(idx);
    auto offset = offsets.at(idx);
    auto stride = strides.at(idx);
    ck::half_t* ptr = reinterpret_cast<ck::half_t*>(ptrs.at(idx));
    ptr += offset;
    offset += stride;
    if (offset == copy * stride) {
        offset = 0;
    }
    offsets[idx] = offset;
    return ptr;
  }
  std::vector<int64_t> offsets;
  std::vector<int64_t> strides;
  std::vector<int64_t> copies;
  std::vector<void*> ptrs;
  std::mt19937 gen;
  std::uniform_int_distribution<int64_t> uniform_dist;
  rocrand_generator generator;
};

// hack for DeviceMem linking error
// TODO fix this by making CK a header-only lib
// <<< hack begin
DeviceMem::DeviceMem(std::size_t mem_size) : mMemSize(mem_size)
{
  hipGetErrorString(hipMalloc(static_cast<void**>(&mpDeviceBuf), mMemSize));
}
void* DeviceMem::GetDeviceBuffer() const { return mpDeviceBuf; }
void DeviceMem::ToDevice(const void* p) const
{
  hipGetErrorString(
        hipMemcpy(mpDeviceBuf, const_cast<void*>(p), mMemSize, hipMemcpyHostToDevice));
}
void DeviceMem::FromDevice(void* p) const
{
  hipGetErrorString(hipMemcpy(p, mpDeviceBuf, mMemSize, hipMemcpyDeviceToHost));
}
DeviceMem::~DeviceMem() { hipGetErrorString(hipFree(mpDeviceBuf)); }
struct KernelTimerImpl
{
  KernelTimerImpl() {
    hipGetErrorString(hipEventCreate(&mStart));
    hipGetErrorString(hipEventCreate(&mEnd));
  }
  ~KernelTimerImpl() {
    hipGetErrorString(hipEventDestroy(mStart));
    hipGetErrorString(hipEventDestroy(mEnd));
  }
  void Start() {
    hipGetErrorString(hipDeviceSynchronize());
    hipGetErrorString(hipEventRecord(mStart, nullptr));
  }
  void End() {
    hipGetErrorString(hipEventRecord(mEnd, nullptr));
    hipGetErrorString(hipEventSynchronize(mEnd));
  }
  float GetElapsedTime() const {
    float time;
    hipGetErrorString(hipEventElapsedTime(&time, mStart, mEnd));
    return time;
  }
  hipEvent_t mStart, mEnd;
};
// >>> hack end


int main(int argc, char** argv) {
  

  const int64_t in_0 = std::stoi(argv[1]);

  const int64_t in_1 = std::stoi(argv[2]);

  auto memory_pool = std::make_unique<ProfilerMemoryPool>();
  hipStream_t stream = nullptr;
  
  int64_t ptr_sz = in_0 * in_1;

  int64_t norm_dim = in_1;

  // TODO: special pool size for 8M L2 cache
  // need to tune it for other devices
  int64_t mem_pool_sz = std::max(2,  std::min(64, int((1 << 23) / ptr_sz)));

  memory_pool->AllocateHalfTensor(ptr_sz, mem_pool_sz);  // in: index 0
  memory_pool->AllocateHalfTensor(ptr_sz, mem_pool_sz);  // out: index 1
  memory_pool->AllocateHalfTensor(norm_dim, mem_pool_sz);  // gamma: index 2
  memory_pool->AllocateHalfTensor(norm_dim, mem_pool_sz);  // beta: index 3

  // warmup
  for(int i = 0; i < 3; ++i) {
    
  layernorm_0(
     (void *) memory_pool->RequestHalfTensorByIdx(0),
     (void *) memory_pool->RequestHalfTensorByIdx(2),
     (void *) memory_pool->RequestHalfTensorByIdx(3),
     (void *) memory_pool->RequestHalfTensorByIdx(1),

      const_cast<int64_t *>(&in_0),

      const_cast<int64_t *>(&in_1),

     stream
  );
    
  }
  // run
  KernelTimerImpl timer;
  timer.Start();
  for(int i = 0; i < 5; ++i) {
    
  layernorm_0(
     (void *) memory_pool->RequestHalfTensorByIdx(0),
     (void *) memory_pool->RequestHalfTensorByIdx(2),
     (void *) memory_pool->RequestHalfTensorByIdx(3),
     (void *) memory_pool->RequestHalfTensorByIdx(1),

      const_cast<int64_t *>(&in_0),

      const_cast<int64_t *>(&in_1),

     stream
  );
    
  }
  timer.End();
  std::cout << "WS:" <<GLOBAL_WORKSPACE_SIZE<<std::endl;
  std::cout << "TIME:" << timer.GetElapsedTime() << std::endl;
}
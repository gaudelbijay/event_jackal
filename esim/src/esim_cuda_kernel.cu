#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_DEVICE(x, y) AT_ASSERTM(x.device().index() == y.device().index(), #x " and " #y " must be in same CUDA device")

// --------------------------------------------------------------------------
// CUDA Kernel: Count events
// --------------------------------------------------------------------------
template <typename scalar_t>
__global__ void count_events_cuda_forward_kernel(
    const scalar_t* __restrict__ imgs,
    const scalar_t* __restrict__ init_refs,
    scalar_t* __restrict__ refs_over_time,
    int64_t* __restrict__ count_ev, 
    int T, int H, int W, float ct_neg, float ct_pos)
{
  const int linIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (linIdx >= H * W)
    return;
    
  scalar_t ref = init_refs[linIdx];
  int tot_num_events = 0;
  
  for (int t = 0; t < T - 1; t++)
  {
    int tidx = (t+1) * H * W + linIdx;
    int tidx_min_1 = t * H * W + linIdx;
    
    scalar_t i0 = imgs[tidx_min_1];
    scalar_t i1 = imgs[tidx];
    
    int polarity = (i1 >= ref) ? 1 : -1;
    float ct = (i1 >= ref) ? ct_pos : ct_neg;
    int num_events = abs(i1 - ref) / ct;
    
    tot_num_events += num_events;
    ref += polarity * ct * num_events;
    
    // Save the updated reference value for later use.
    refs_over_time[tidx_min_1] = ref;
  }
  count_ev[linIdx] = tot_num_events;
}

// --------------------------------------------------------------------------
// CUDA Kernel: Generate events
// --------------------------------------------------------------------------
template <typename scalar_t>
__global__ void esim_cuda_forward_kernel(
  const scalar_t* __restrict__ imgs,
  const int64_t* __restrict__ ts,
  const scalar_t* __restrict__ init_ref,
  const scalar_t* __restrict__ refs_over_time,
  const int64_t* __restrict__ offsets,
  int64_t* __restrict__ ev,
  int64_t* __restrict__ t_last_ev,
  int T, int H, int W, float ct_neg, float ct_pos, int64_t t_ref)
{
  const int linIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (linIdx >= H * W)
    return;
    
  int x = linIdx % W;
  int y = linIdx / W;
  
  scalar_t ref0 = init_ref[linIdx];
  int64_t offset = offsets[linIdx];
  
  for (int t = 0; t < T - 1; t++) {
    scalar_t i0 = imgs[linIdx + t * H * W];
    scalar_t i1 = imgs[linIdx + (t + 1) * H * W];
    
    int64_t t0 = ts[t];
    int64_t t1 = ts[t + 1];
    
    if (t > 0) {
      ref0 = refs_over_time[linIdx + (t - 1) * H * W];
    }
    
    int polarity = (i1 >= ref0) ? 1 : -1;
    float ct = (i1 >= ref0) ? ct_pos : ct_neg;
    int64_t num_events = abs(i1 - ref0) / ct;
    
    int64_t t_prev = t_last_ev[linIdx];
    for (int evIdx = 0; evIdx < num_events; evIdx++) 
    {
      scalar_t r = (ref0 + (evIdx + 1) * polarity * ct - i0) / (i1 - i0);
      int64_t timestamp = t0 + (t1 - t0) * r;
      int64_t delta_t = timestamp - t_prev;
      
      if (delta_t > t_ref || t_prev == 0) {
          int64_t idx = 4 * (offset + evIdx);
          ev[idx + 0] = x;
          ev[idx + 1] = y;
          ev[idx + 2] = timestamp;
          ev[idx + 3] = polarity;
          t_last_ev[linIdx] = timestamp;
          t_prev = timestamp;
      }
    }
    offset += num_events;
  }
}

// --------------------------------------------------------------------------
// Host functions wrapping CUDA kernel launches
// --------------------------------------------------------------------------
std::vector<torch::Tensor> esim_forward_count_events(
  const torch::Tensor& imgs,
  const torch::Tensor& init_refs,
  torch::Tensor& refs_over_time,
  torch::Tensor& count_ev,
  float ct_neg,
  float ct_pos)
{
  CHECK_INPUT(imgs);
  CHECK_INPUT(count_ev);
  CHECK_INPUT(init_refs);
  CHECK_INPUT(refs_over_time);
  
  CHECK_DEVICE(imgs, count_ev);
  CHECK_DEVICE(imgs, init_refs);
  CHECK_DEVICE(imgs, refs_over_time);
  
  unsigned T = imgs.size(0);
  unsigned H = imgs.size(1);
  unsigned W = imgs.size(2);
  
  unsigned threads = 256;
  dim3 blocks((H * W + threads - 1) / threads, 1);
  
  count_events_cuda_forward_kernel<float><<<blocks, threads>>>(
      imgs.data_ptr<float>(), 
      init_refs.data_ptr<float>(),
      refs_over_time.data_ptr<float>(),
      count_ev.data_ptr<int64_t>(),
      T, H, W, ct_neg, ct_pos
  );
  
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }
  
  return {refs_over_time, count_ev};
}

torch::Tensor esim_forward(
    const torch::Tensor& imgs,
    const torch::Tensor& ts,
    const torch::Tensor& init_refs,
    const torch::Tensor& refs_over_time,
    const torch::Tensor& offsets,
    torch::Tensor& ev,
    torch::Tensor& t_last_ev,
    float ct_neg,
    float ct_pos,
    int64_t dt_ref)
{
  CHECK_INPUT(imgs);
  CHECK_INPUT(ts);
  CHECK_INPUT(ev);
  CHECK_INPUT(offsets);
  CHECK_INPUT(refs_over_time);
  CHECK_INPUT(init_refs);
  
  CHECK_DEVICE(imgs, ts);
  CHECK_DEVICE(imgs, ev);
  CHECK_DEVICE(imgs, offsets);
  CHECK_DEVICE(imgs, init_refs);
  CHECK_DEVICE(imgs, refs_over_time);
  CHECK_DEVICE(imgs, t_last_ev);
  
  unsigned T = imgs.size(0);
  unsigned H = imgs.size(1);
  unsigned W = imgs.size(2);
  
  unsigned threads = 256;
  dim3 blocks((H * W + threads - 1) / threads, 1);
  
  esim_cuda_forward_kernel<float><<<blocks, threads>>>(
      imgs.data_ptr<float>(),
      ts.data_ptr<int64_t>(), 
      init_refs.data_ptr<float>(),
      refs_over_time.data_ptr<float>(),
      offsets.data_ptr<int64_t>(),
      ev.data_ptr<int64_t>(),
      t_last_ev.data_ptr<int64_t>(),
      T, H, W, ct_neg, ct_pos, dt_ref
  );
  
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }
  
  return ev;
}

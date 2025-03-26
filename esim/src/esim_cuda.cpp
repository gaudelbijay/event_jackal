
#include <torch/extension.h>
#include <vector>

// CUDA forward declarations

torch::Tensor esim_forward(
    const torch::Tensor& imgs,      // T x H x W
    const torch::Tensor& ts,        // T
    const torch::Tensor& init_refs, // H x W
    const torch::Tensor& refs_over_time, // T-1 x H x W
    const torch::Tensor& offsets,   // H x W
    torch::Tensor& ev,              // N x 4 (x, y, t, p)
    torch::Tensor& t_last_ev,       // H x W
    float ct_neg,
    float ct_pos,
    int64_t dt_ref);

// Match the .cu "esim_forward_count_events" function exactly:
std::vector<torch::Tensor> esim_forward_count_events(
    const torch::Tensor& imgs,        // T x H x W
    const torch::Tensor& init_refs,   // H x W
    torch::Tensor& refs_over_time,    // T-1 x H x W
    torch::Tensor& count_ev,          // H x W
    float ct_neg,
    float ct_pos);


#include <cstdlib>
#include <stdexcept>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>

#include <ATen/ATen.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"

#include "aclrtlaunch_group_dynamic_quant_custom.h"
#include "aclrtlaunch_quant_group_dynamic_pack_aic_read_custom.h"
#include "aclrtlaunch_quant_group_dynamic_pack_custom.h"
#include "aclrtlaunch_quant_group_tile_workspace_roundtrip_custom.h"
#include "aclrtlaunch_quant_group_tile_matmul_compare_custom.h"
#include "aclrtlaunch_quant_group_tile_dynamic_pack_debug_custom.h"
#include "aclrtlaunch_quant_group_shared_tile_pack_debug_custom.h"
#include "aclrtlaunch_quant_group_matmul_passthrough_dynamic_custom.h"
#include "aclrtlaunch_quant_group_matmul_workspace_passthrough_dynamic_custom.h"
#include "aclrtlaunch_quant_group_matmul_external_dynamic_custom.h"
#include "aclrtlaunch_quant_group_matmul_tile_dynamic_custom.h"
#include "aclrtlaunch_quant_group_matmul_dynamic_custom.h"
#include "aclrtlaunch_quant_group_matmul_custom.h"
#include "aclrtlaunch_quant_group_matmul_custom_int8.h"
#include "../quant_group_matmul_custom_tiling.h"

extern bool GenerateTiling(QuantGroupMatmulCustomTilingData& gmmTiling);

namespace {

struct CacheKey {
  int device_index;
  int64_t m;
  int64_t k;
  int64_t n;
  int64_t group_num;
  int64_t group_axis;
  int64_t k_group_size;
  int64_t k_group_num;
  int64_t block_dim;
  int64_t parall_num;
  int64_t tile_dynamic_schedule;

  bool operator==(const CacheKey& other) const {
    return device_index == other.device_index &&
           m == other.m &&
           k == other.k &&
           n == other.n &&
           group_num == other.group_num &&
           group_axis == other.group_axis &&
           k_group_size == other.k_group_size &&
           k_group_num == other.k_group_num &&
           block_dim == other.block_dim &&
           parall_num == other.parall_num &&
           tile_dynamic_schedule == other.tile_dynamic_schedule;
  }
};

struct CacheKeyHash {
  size_t operator()(const CacheKey& key) const {
    size_t h = std::hash<int>{}(key.device_index);
    h = h * 1315423911u + std::hash<int64_t>{}(key.m);
    h = h * 1315423911u + std::hash<int64_t>{}(key.k);
    h = h * 1315423911u + std::hash<int64_t>{}(key.n);
    h = h * 1315423911u + std::hash<int64_t>{}(key.group_num);
    h = h * 1315423911u + std::hash<int64_t>{}(key.group_axis);
    h = h * 1315423911u + std::hash<int64_t>{}(key.k_group_size);
    h = h * 1315423911u + std::hash<int64_t>{}(key.k_group_num);
    h = h * 1315423911u + std::hash<int64_t>{}(key.block_dim);
    h = h * 1315423911u + std::hash<int64_t>{}(key.parall_num);
    h = h * 1315423911u + std::hash<int64_t>{}(key.tile_dynamic_schedule);
    return h;
  }
};

struct CachedLaunchTensors {
  at::Tensor group_list;
  at::Tensor tiling_npu;
};

struct GroupDynamicQuantTilingData {
  uint32_t m;
  uint32_t k;
  uint32_t group_size;
  uint32_t group_num;
  uint32_t core_num;
};

std::unordered_map<CacheKey, CachedLaunchTensors, CacheKeyHash> g_launch_tensor_cache;
std::mutex g_launch_tensor_cache_mutex;

void check_npu(const at::Tensor& t, const char* name) {
  if (!t.device().is_privateuseone()) {
    throw std::runtime_error(std::string(name) + " must be on NPU device.");
  }
}

size_t align_up(size_t value, size_t align) {
  return (value + align - 1) / align * align;
}

constexpr size_t kDynamicWorkspaceGuardBytes = 512;

uint32_t read_env_uint(const char* name, uint32_t default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }
  char* end = nullptr;
  unsigned long parsed = std::strtoul(value, &end, 10);
  if (end == value || *end != '\0' || parsed == 0) {
    return default_value;
  }
  return static_cast<uint32_t>(parsed);
}

uint32_t read_tile_dynamic_schedule() {
  const char* value = std::getenv("GROUP_QUANT_TILE_DYNAMIC_SCHEDULE");
  if (value == nullptr || value[0] == '\0') {
    return 0U;
  }
  std::string schedule(value);
  if (schedule == "shared_m_outer" || schedule == "1") {
    return 1U;
  }
  return 0U;
}

at::Tensor pack_activation_k_grouped(
    const at::Tensor& x,
    int64_t group_size) {
  check_npu(x, "x");
  TORCH_CHECK(x.dim() == 2, "x must be 2D [M, K].");
  TORCH_CHECK(
      x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
      "x must be float16 or bfloat16.");
  TORCH_CHECK(group_size > 0, "group_size must be positive.");

  const int64_t m = x.size(0);
  const int64_t k = x.size(1);
  TORCH_CHECK(k % group_size == 0, "K must be divisible by group_size.");
  const int64_t k_group_num = k / group_size;

  return x.contiguous()
      .view({m, k_group_num, group_size})
      .permute({1, 0, 2})
      .contiguous()
      .view({k_group_num * m, group_size});
}

std::tuple<at::Tensor, at::Tensor> dynamic_quant_k_grouped_ascendc(
    const at::Tensor& x,
    int64_t group_size,
    int64_t block_dim = 8) {
  check_npu(x, "x");
  TORCH_CHECK(x.dim() == 2, "x must be 2D [M, K].");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "AscendC prototype currently supports bf16 input only.");
  TORCH_CHECK(group_size > 0, "group_size must be positive.");

  const int64_t m = x.size(0);
  const int64_t k = x.size(1);
  TORCH_CHECK(k % group_size == 0, "K must be divisible by group_size.");
  const int64_t group_num = k / group_size;
  TORCH_CHECK(m <= std::numeric_limits<uint32_t>::max(), "M is too large for tiling.");
  TORCH_CHECK(k <= std::numeric_limits<uint32_t>::max(), "K is too large for tiling.");
  TORCH_CHECK(group_size <= std::numeric_limits<uint32_t>::max(), "group_size is too large for tiling.");
  TORCH_CHECK(group_num <= std::numeric_limits<uint32_t>::max(), "group_num is too large for tiling.");

  auto x_q = at::empty({group_num * m, group_size}, x.options().dtype(at::kChar));
  auto scale = at::empty({group_num, m}, x.options().dtype(at::kFloat));

  GroupDynamicQuantTilingData tiling{
      static_cast<uint32_t>(m),
      static_cast<uint32_t>(k),
      static_cast<uint32_t>(group_size),
      static_cast<uint32_t>(group_num),
      static_cast<uint32_t>(block_dim),
  };
  auto tiling_cpu = at::from_blob(
                        &tiling,
                        {static_cast<long>(sizeof(GroupDynamicQuantTilingData))},
                        at::TensorOptions().dtype(at::kByte))
                        .clone();
  auto tiling_npu = tiling_cpu.to(
      c10::Device(c10::DeviceType::PrivateUse1, x.device().index()),
      /*non_blocking=*/false);

  auto workspace = at::empty({1}, x.options().dtype(at::kByte));
  auto stream = c10_npu::getCurrentNPUStream().stream();
  uint32_t ret = aclrtlaunch_group_dynamic_quant_custom(
      static_cast<uint32_t>(block_dim),
      stream,
      const_cast<void*>(x.data_ptr()),
      x_q.data_ptr(),
      scale.data_ptr(),
      workspace.data_ptr(),
      tiling_npu.data_ptr());
  TORCH_CHECK(ret == 0, "aclrtlaunch_group_dynamic_quant_custom failed, ret=", ret);
  return {x_q, scale};
}

CachedLaunchTensors get_cached_launch_tensors(
    int device_index,
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t group_num,
    int64_t group_axis,
    int64_t k_group_size,
    int64_t k_group_num,
    int64_t block_dim,
    int64_t parall_num) {
  const uint32_t tile_dynamic_schedule = read_tile_dynamic_schedule();
  CacheKey key{
      device_index,
      m,
      k,
      n,
      group_num,
      group_axis,
      k_group_size,
      k_group_num,
      block_dim,
      parall_num,
      static_cast<int64_t>(tile_dynamic_schedule),
  };
  {
    std::lock_guard<std::mutex> guard(g_launch_tensor_cache_mutex);
    auto it = g_launch_tensor_cache.find(key);
    if (it != g_launch_tensor_cache.end()) {
      return it->second;
    }
  }

  QuantGroupMatmulCustomTilingData t{};
  t.coreNum = static_cast<uint32_t>(block_dim);
  t.groupNum = static_cast<uint32_t>(group_num);
  t.totalInGroup = static_cast<uint32_t>(m);
  t.k = static_cast<uint32_t>(k);
  t.n = static_cast<uint32_t>(n);
  t.groupAxis = static_cast<uint32_t>(group_axis);
  t.kGroupSize = static_cast<uint32_t>(k_group_size);
  t.kGroupNum = static_cast<uint32_t>(k_group_num);
  t.ubCalSize = 16 * 256;
  t.ubRestBytes = 118784U;
  t.parallNum = static_cast<uint32_t>(parall_num);
  t.enableTilePipeline =
      std::getenv("GROUP_QUANT_ENABLE_TILE_PIPELINE") != nullptr ? 1U : 0U;
  t.tileDynamicSchedule = tile_dynamic_schedule;
  TORCH_CHECK(GenerateTiling(t), "GenerateTiling failed.");

  auto tiling_cpu = at::from_blob(
                        &t,
                        {static_cast<long>(sizeof(QuantGroupMatmulCustomTilingData))},
                        at::TensorOptions().dtype(at::kByte))
                        .clone();
  auto tiling_npu = tiling_cpu.to(
      c10::Device(c10::DeviceType::PrivateUse1, device_index),
      /*non_blocking=*/false);
  auto group_list_cpu = at::full(
      {1},
      m,
      at::TensorOptions().dtype(at::kLong));
  auto group_list_npu = group_list_cpu.to(
      c10::Device(c10::DeviceType::PrivateUse1, device_index),
      /*non_blocking=*/false);

  CachedLaunchTensors cached{
      std::move(group_list_npu),
      std::move(tiling_npu),
  };
  std::lock_guard<std::mutex> guard(g_launch_tensor_cache_mutex);
  auto [it, inserted] = g_launch_tensor_cache.emplace(key, cached);
  if (!inserted) {
    return it->second;
  }
  return cached;
}

}  // namespace

at::Tensor quant_group_matmul_forward_grouped(
    const at::Tensor& x_int8,
    const at::Tensor& weight_nz_int8,
    const at::Tensor& scale_f32,
    const at::Tensor& pertoken_scale_f32,
    const at::Tensor& group_list,
    int64_t group_axis,
    int64_t k_group_size,
    int64_t k_group_num,
    int64_t block_dim,
    int64_t parall_num);

void check_common_inputs(
    const at::Tensor& x_int8,
    const at::Tensor& weight_nz_int8,
    const at::Tensor& scale_f32,
    const at::Tensor& pertoken_scale_f32,
    const at::Tensor& group_list) {
  TORCH_CHECK(x_int8.scalar_type() == at::kChar, "x_int8 must be int8.");
  TORCH_CHECK(weight_nz_int8.scalar_type() == at::kChar, "weight_nz_int8 must be int8.");
  TORCH_CHECK(scale_f32.scalar_type() == at::kFloat, "scale_f32 must be float32.");
  TORCH_CHECK(pertoken_scale_f32.scalar_type() == at::kFloat, "pertoken_scale_f32 must be float32.");
  TORCH_CHECK(group_list.scalar_type() == at::kLong, "group_list must be int64.");
  TORCH_CHECK(x_int8.dim() == 2, "x_int8 must be 2D.");
  TORCH_CHECK(scale_f32.dim() == 2, "scale_f32 must be 2D [group_num, N].");
  TORCH_CHECK(pertoken_scale_f32.dim() == 1, "pertoken_scale_f32 must be 1D.");
  TORCH_CHECK(group_list.dim() == 1, "group_list must be 1D [group_num].");

  check_npu(x_int8, "x_int8");
  check_npu(weight_nz_int8, "weight_nz_int8");
  check_npu(scale_f32, "scale_f32");
  check_npu(pertoken_scale_f32, "pertoken_scale_f32");
  check_npu(group_list, "group_list");
}

at::Tensor quant_group_matmul_forward_grouped(
    const at::Tensor& x_int8,
    const at::Tensor& weight_nz_int8,
    const at::Tensor& scale_f32,
    const at::Tensor& pertoken_scale_f32,
    const at::Tensor& group_list,
    int64_t group_axis = 0,
    int64_t k_group_size = 0,
    int64_t k_group_num = 0,
    int64_t block_dim = 8,
    int64_t parall_num = 4) {
  check_common_inputs(
      x_int8,
      weight_nz_int8,
      scale_f32,
      pertoken_scale_f32,
      group_list);

  const int64_t x_m = x_int8.size(0);
  const int64_t x_k = x_int8.size(1);
  const int64_t n = scale_f32.size(1);
  TORCH_CHECK(group_axis == 0 || group_axis == 1 || group_axis == 2, "group_axis must be 0, 1 or 2.");
  int64_t logical_group_num = group_list.size(0);
  const int64_t logical_m = group_list.sum().item<int64_t>();
  int64_t logical_k = x_k;
  if (group_axis == 0) {
    TORCH_CHECK(pertoken_scale_f32.size(0) == x_m, "pertoken_scale_f32 size mismatch with M.");
    TORCH_CHECK(group_list.sum().item<int64_t>() == x_m, "sum(group_list) must equal M.");
    TORCH_CHECK(
        scale_f32.size(0) == logical_group_num,
        "scale_f32 group dimension mismatch for M-axis grouping.");
  }
  if (group_axis == 1) {
    TORCH_CHECK(k_group_size > 0, "k_group_size must be > 0 for K-axis grouping.");
    TORCH_CHECK(k_group_num > 0, "k_group_num must be > 0 for K-axis grouping.");
    TORCH_CHECK(
        group_list.size(0) == 1,
        "group_list must have a single entry [M] for K-axis grouping.");
    TORCH_CHECK(
        scale_f32.size(0) == k_group_num,
        "scale_f32 must have shape [k_group_num, N] for K-axis grouping.");
    TORCH_CHECK(
        x_k == k_group_size,
        "x_int8 second dim must equal k_group_size for K-axis grouping.");
    TORCH_CHECK(
        x_m == logical_m * k_group_num,
        "x_int8 first dim must equal M * k_group_num for K-axis grouping.");
    TORCH_CHECK(
        pertoken_scale_f32.size(0) == logical_m * k_group_num,
        "pertoken_scale_f32 must have shape [k_group_num * M] for K-axis grouping.");
    logical_k = k_group_size * k_group_num;
    logical_group_num = k_group_num;
  }
  if (group_axis == 2) {
    TORCH_CHECK(k_group_size > 0, "k_group_size must be > 0 for expert x K-axis grouping.");
    TORCH_CHECK(k_group_num > 0, "k_group_num must be > 0 for expert x K-axis grouping.");
    TORCH_CHECK(
        group_list.size(0) > 0,
        "group_list must contain expert token counts for expert x K-axis grouping.");
    TORCH_CHECK(
        scale_f32.size(0) == logical_group_num * k_group_num,
        "scale_f32 must have shape [expert_num * k_group_num, N] for expert x K-axis grouping.");
    TORCH_CHECK(
        x_k == k_group_size,
        "x_int8 second dim must equal k_group_size for expert x K-axis grouping.");
    TORCH_CHECK(
        x_m == logical_m * k_group_num,
        "x_int8 first dim must equal M * k_group_num for expert x K-axis grouping.");
    TORCH_CHECK(
        pertoken_scale_f32.size(0) == logical_m * k_group_num,
        "pertoken_scale_f32 must have shape [k_group_num * M] for expert x K-axis grouping.");
    logical_k = k_group_size * k_group_num;
  }

  const int64_t tiling_k = (group_axis == 1 || group_axis == 2) ? k_group_size : logical_k;
  auto cached = get_cached_launch_tensors(
      x_int8.device().index(),
      logical_m,
      tiling_k,
      n,
      logical_group_num,
      group_axis,
      k_group_size,
      k_group_num,
      block_dim,
      parall_num);
  auto y = at::empty({logical_m, n}, x_int8.options().dtype(at::kHalf));

  auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
  const size_t system_workspace =
      static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
  const size_t mm_workspace_groups =
      (group_axis == 1 || group_axis == 2) ? static_cast<size_t>(k_group_num) : 1ULL;
  const size_t user_workspace =
      static_cast<size_t>(parall_num) * 256ULL * 128ULL * sizeof(int32_t) *
      static_cast<size_t>(block_dim) * mm_workspace_groups;
  const bool keep_acc_workspace =
      std::getenv("GROUP_QUANT_KEEP_ACC_WORKSPACE") != nullptr;
  const size_t acc_workspace =
      (keep_acc_workspace && (group_axis == 1 || group_axis == 2))
          ? static_cast<size_t>(logical_m) * static_cast<size_t>(n) * sizeof(float)
          : 0;
  const size_t workspace_size = user_workspace + acc_workspace + system_workspace;
  auto workspace = at::empty(
      {static_cast<long>(workspace_size)}, x_int8.options().dtype(at::kByte));

  auto stream = c10_npu::getCurrentNPUStream().stream();
  auto* bias_ptr = static_cast<void*>(nullptr);  // bias unused in this kernel path
  uint32_t ret = aclrtlaunch_quant_group_matmul_custom(
      static_cast<uint32_t>(block_dim),
      stream,
      const_cast<void*>(x_int8.data_ptr()),
      const_cast<void*>(weight_nz_int8.data_ptr()),
      bias_ptr,
      const_cast<void*>(group_list.data_ptr()),
      const_cast<void*>(scale_f32.data_ptr()),
      const_cast<void*>(pertoken_scale_f32.data_ptr()),
      y.data_ptr(),
      workspace.data_ptr(),
      cached.tiling_npu.data_ptr());
  TORCH_CHECK(ret == 0, "aclrtlaunch_quant_group_matmul_custom failed, ret=", ret);
  return y;
}

at::Tensor quant_group_matmul_forward_k_grouped(
    const at::Tensor& x_int8,
    const at::Tensor& weight_nz_int8,
    const at::Tensor& scale_f32,
    const at::Tensor& pertoken_scale_f32,
    int64_t k_group_size,
    int64_t k_group_num,
    int64_t block_dim = 8,
    int64_t parall_num = 4) {
  TORCH_CHECK(x_int8.scalar_type() == at::kChar, "x_int8 must be int8.");
  TORCH_CHECK(weight_nz_int8.scalar_type() == at::kChar, "weight_nz_int8 must be int8.");
  TORCH_CHECK(scale_f32.scalar_type() == at::kFloat, "scale_f32 must be float32.");
  TORCH_CHECK(pertoken_scale_f32.scalar_type() == at::kFloat, "pertoken_scale_f32 must be float32.");
  TORCH_CHECK(x_int8.dim() == 2, "x_int8 must be 2D.");
  TORCH_CHECK(scale_f32.dim() == 2, "scale_f32 must be 2D [k_group_num, N].");
  TORCH_CHECK(pertoken_scale_f32.dim() == 1, "pertoken_scale_f32 must be 1D.");
  check_npu(x_int8, "x_int8");
  check_npu(weight_nz_int8, "weight_nz_int8");
  check_npu(scale_f32, "scale_f32");
  check_npu(pertoken_scale_f32, "pertoken_scale_f32");

  TORCH_CHECK(k_group_size > 0, "k_group_size must be > 0.");
  TORCH_CHECK(k_group_num > 0, "k_group_num must be > 0.");
  const int64_t x_m = x_int8.size(0);
  const int64_t x_k = x_int8.size(1);
  const int64_t n = scale_f32.size(1);
  TORCH_CHECK(x_k == k_group_size, "x_int8 second dim must equal k_group_size.");
  TORCH_CHECK(x_m % k_group_num == 0, "x_int8 first dim must be divisible by k_group_num.");
  TORCH_CHECK(scale_f32.size(0) == k_group_num, "scale_f32 first dim must equal k_group_num.");
  TORCH_CHECK(pertoken_scale_f32.size(0) == x_m, "pertoken_scale_f32 size must equal x_int8 first dim.");
  const int64_t logical_m = x_m / k_group_num;

  auto cached = get_cached_launch_tensors(
      x_int8.device().index(),
      logical_m,
      k_group_size,
      n,
      k_group_num,
      1,
      k_group_size,
      k_group_num,
      block_dim,
      parall_num);
  auto y = at::empty({logical_m, n}, x_int8.options().dtype(at::kHalf));

  auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
  const size_t system_workspace =
      static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
  const size_t user_workspace =
      static_cast<size_t>(parall_num) * 256ULL * 128ULL * sizeof(int32_t) *
      static_cast<size_t>(block_dim) * static_cast<size_t>(k_group_num);
  const bool keep_acc_workspace =
      std::getenv("GROUP_QUANT_KEEP_ACC_WORKSPACE") != nullptr;
  const size_t acc_workspace =
      keep_acc_workspace
          ? static_cast<size_t>(logical_m) * static_cast<size_t>(n) * sizeof(float)
          : 0;
  const size_t workspace_size = user_workspace + acc_workspace + system_workspace;
  auto workspace = at::empty(
      {static_cast<long>(workspace_size)}, x_int8.options().dtype(at::kByte));

  auto stream = c10_npu::getCurrentNPUStream().stream();
  auto* bias_ptr = static_cast<void*>(nullptr);
  uint32_t ret = aclrtlaunch_quant_group_matmul_custom(
      static_cast<uint32_t>(block_dim),
      stream,
      const_cast<void*>(x_int8.data_ptr()),
      const_cast<void*>(weight_nz_int8.data_ptr()),
      bias_ptr,
      cached.group_list.data_ptr(),
      const_cast<void*>(scale_f32.data_ptr()),
      const_cast<void*>(pertoken_scale_f32.data_ptr()),
      y.data_ptr(),
      workspace.data_ptr(),
      cached.tiling_npu.data_ptr());
  TORCH_CHECK(ret == 0, "aclrtlaunch_quant_group_matmul_custom failed, ret=", ret);
  return y;
}

at::Tensor quant_group_matmul_forward_k_grouped_dynamic_fused(
    const at::Tensor& x,
    const at::Tensor& weight_nz_int8,
    const at::Tensor& scale_f32,
    int64_t k_group_size,
    int64_t block_dim = 8,
    int64_t parall_num = 4) {
  check_npu(x, "x");
  check_npu(weight_nz_int8, "weight_nz_int8");
  check_npu(scale_f32, "scale_f32");
  TORCH_CHECK(x.dim() == 2, "x must be 2D [M, K].");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "single-kernel dynamic prototype supports bf16 input only.");
  TORCH_CHECK(weight_nz_int8.scalar_type() == at::kChar, "weight_nz_int8 must be int8.");
  TORCH_CHECK(scale_f32.scalar_type() == at::kFloat, "scale_f32 must be float32.");
  TORCH_CHECK(scale_f32.dim() == 2, "scale_f32 must be 2D [k_group_num, N].");
  TORCH_CHECK(k_group_size > 0, "k_group_size must be > 0.");
  TORCH_CHECK(x.size(1) % k_group_size == 0, "K must be divisible by k_group_size.");

  const int64_t logical_m = x.size(0);
  const int64_t k_group_num = x.size(1) / k_group_size;
  const int64_t n = scale_f32.size(1);
  TORCH_CHECK(scale_f32.size(0) == k_group_num, "scale_f32 first dim must equal K group count.");

  auto cached = get_cached_launch_tensors(
      x.device().index(),
      logical_m,
      k_group_size,
      n,
      k_group_num,
      1,
      k_group_size,
      k_group_num,
      block_dim,
      parall_num);
  auto y = at::empty({logical_m, n}, x.options().dtype(at::kHalf));

  auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
  const size_t system_workspace =
      static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
  const size_t x_q_workspace =
      static_cast<size_t>(logical_m) * static_cast<size_t>(k_group_num) *
      static_cast<size_t>(k_group_size) * sizeof(int8_t);
  const size_t pertoken_workspace =
      static_cast<size_t>(logical_m) * static_cast<size_t>(k_group_num) * sizeof(float);
  const size_t x_q_workspace_aligned = align_up(x_q_workspace, 32);
  const size_t pertoken_workspace_aligned = align_up(pertoken_workspace, 32);
  const size_t user_workspace =
      static_cast<size_t>(parall_num) * 256ULL * 128ULL * sizeof(int32_t) *
      static_cast<size_t>(block_dim) * static_cast<size_t>(k_group_num);
  const bool keep_acc_workspace =
      std::getenv("GROUP_QUANT_KEEP_ACC_WORKSPACE") != nullptr;
  const size_t acc_workspace =
      keep_acc_workspace
          ? static_cast<size_t>(logical_m) * static_cast<size_t>(n) * sizeof(float)
          : 0;
  const size_t x_q_workspace_offset = align_up(kDynamicWorkspaceGuardBytes, 512);
  const size_t pertoken_workspace_offset =
      align_up(x_q_workspace_offset + x_q_workspace_aligned, 512);
  const size_t matmul_workspace_offset =
      align_up(pertoken_workspace_offset + pertoken_workspace_aligned, 512);
  const size_t workspace_size =
      matmul_workspace_offset + user_workspace + acc_workspace + system_workspace;
  auto workspace = at::empty(
      {static_cast<long>(workspace_size)}, x.options().dtype(at::kByte));

  auto stream = c10_npu::getCurrentNPUStream().stream();
  auto* bias_ptr = static_cast<void*>(nullptr);
  uint32_t ret = aclrtlaunch_quant_group_matmul_dynamic_custom(
      static_cast<uint32_t>(block_dim),
      stream,
      const_cast<void*>(x.data_ptr()),
      const_cast<void*>(weight_nz_int8.data_ptr()),
      bias_ptr,
      cached.group_list.data_ptr(),
      const_cast<void*>(scale_f32.data_ptr()),
      y.data_ptr(),
      workspace.data_ptr(),
      cached.tiling_npu.data_ptr());
  TORCH_CHECK(ret == 0, "aclrtlaunch_quant_group_matmul_dynamic_custom failed, ret=", ret);
  return y;
}

at::Tensor quant_group_matmul_forward_k_grouped_dynamic_external_fused(
    const at::Tensor& x,
    const at::Tensor& weight_nz_int8,
    const at::Tensor& scale_f32,
    int64_t k_group_size,
    int64_t block_dim = 8,
    int64_t parall_num = 4) {
  check_npu(x, "x");
  check_npu(weight_nz_int8, "weight_nz_int8");
  check_npu(scale_f32, "scale_f32");
  TORCH_CHECK(x.dim() == 2, "x must be 2D [M, K].");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "external dynamic prototype supports bf16 input only.");
  TORCH_CHECK(weight_nz_int8.scalar_type() == at::kChar, "weight_nz_int8 must be int8.");
  TORCH_CHECK(scale_f32.scalar_type() == at::kFloat, "scale_f32 must be float32.");
  TORCH_CHECK(scale_f32.dim() == 2, "scale_f32 must be 2D [k_group_num, N].");
  TORCH_CHECK(k_group_size > 0, "k_group_size must be > 0.");
  TORCH_CHECK(x.size(1) % k_group_size == 0, "K must be divisible by k_group_size.");

  const int64_t logical_m = x.size(0);
  const int64_t k_group_num = x.size(1) / k_group_size;
  const int64_t n = scale_f32.size(1);
  TORCH_CHECK(scale_f32.size(0) == k_group_num, "scale_f32 first dim must equal K group count.");

  auto cached = get_cached_launch_tensors(
      x.device().index(),
      logical_m,
      k_group_size,
      n,
      k_group_num,
      1,
      k_group_size,
      k_group_num,
      block_dim,
      parall_num);
  auto x_q = at::empty(
      {logical_m * k_group_num, k_group_size},
      x.options().dtype(at::kChar));
  auto pertoken_scale = at::empty(
      {logical_m * k_group_num},
      x.options().dtype(at::kFloat));
  auto y = at::empty({logical_m, n}, x.options().dtype(at::kHalf));

  auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
  const size_t system_workspace =
      static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
  const size_t user_workspace =
      static_cast<size_t>(parall_num) * 256ULL * 128ULL * sizeof(int32_t) *
      static_cast<size_t>(block_dim) * static_cast<size_t>(k_group_num);
  const bool keep_acc_workspace =
      std::getenv("GROUP_QUANT_KEEP_ACC_WORKSPACE") != nullptr;
  const size_t acc_workspace =
      keep_acc_workspace
          ? static_cast<size_t>(logical_m) * static_cast<size_t>(n) * sizeof(float)
          : 0;
  const size_t workspace_size = user_workspace + acc_workspace + system_workspace;
  auto workspace = at::empty(
      {static_cast<long>(workspace_size)}, x.options().dtype(at::kByte));

  auto stream = c10_npu::getCurrentNPUStream().stream();
  auto* bias_ptr = static_cast<void*>(nullptr);
  uint32_t ret = aclrtlaunch_quant_group_matmul_external_dynamic_custom(
      static_cast<uint32_t>(block_dim),
      stream,
      const_cast<void*>(x.data_ptr()),
      const_cast<void*>(weight_nz_int8.data_ptr()),
      bias_ptr,
      cached.group_list.data_ptr(),
      const_cast<void*>(scale_f32.data_ptr()),
      x_q.data_ptr(),
      pertoken_scale.data_ptr(),
      y.data_ptr(),
      workspace.data_ptr(),
      cached.tiling_npu.data_ptr());
  TORCH_CHECK(ret == 0, "aclrtlaunch_quant_group_matmul_external_dynamic_custom failed, ret=", ret);
  return y;
}

at::Tensor quant_group_matmul_forward_k_grouped_tile_dynamic(
    const at::Tensor& x,
    const at::Tensor& weight_nz_int8,
    const at::Tensor& scale_f32,
    int64_t k_group_size,
    int64_t block_dim = 8,
    int64_t parall_num = 4) {
  check_npu(x, "x");
  check_npu(weight_nz_int8, "weight_nz_int8");
  check_npu(scale_f32, "scale_f32");
  TORCH_CHECK(x.dim() == 2, "x must be 2D [M, K].");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "tile dynamic prototype supports bf16 input only.");
  TORCH_CHECK(weight_nz_int8.scalar_type() == at::kChar, "weight_nz_int8 must be int8.");
  TORCH_CHECK(scale_f32.scalar_type() == at::kFloat, "scale_f32 must be float32.");
  TORCH_CHECK(scale_f32.dim() == 2, "scale_f32 must be 2D [k_group_num, N].");
  TORCH_CHECK(k_group_size > 0, "k_group_size must be > 0.");
  TORCH_CHECK(x.size(1) % k_group_size == 0, "K must be divisible by k_group_size.");

  const int64_t logical_m = x.size(0);
  const int64_t k_group_num = x.size(1) / k_group_size;
  const int64_t n = scale_f32.size(1);
  TORCH_CHECK(scale_f32.size(0) == k_group_num, "scale_f32 first dim must equal K group count.");

  auto cached = get_cached_launch_tensors(
      x.device().index(),
      logical_m,
      k_group_size,
      n,
      k_group_num,
      1,
      k_group_size,
      k_group_num,
      block_dim,
      parall_num);
  auto x_q = at::empty(
      {logical_m * k_group_num, k_group_size},
      x.options().dtype(at::kChar));
  auto pertoken_scale = at::empty(
      {logical_m * k_group_num},
      x.options().dtype(at::kFloat));
  auto y = at::empty({logical_m, n}, x.options().dtype(at::kHalf));

  auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
  const size_t system_workspace =
      static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
  const size_t user_workspace =
      static_cast<size_t>(parall_num) * 256ULL * 128ULL * sizeof(int32_t) *
      static_cast<size_t>(block_dim) * static_cast<size_t>(k_group_num);
  const bool keep_acc_workspace =
      std::getenv("GROUP_QUANT_KEEP_ACC_WORKSPACE") != nullptr;
  const size_t acc_workspace =
      keep_acc_workspace
          ? static_cast<size_t>(logical_m) * static_cast<size_t>(n) * sizeof(float)
          : 0;
  const size_t tile_acc_workspace =
      static_cast<size_t>(logical_m) * static_cast<size_t>(n) * sizeof(float);
  const size_t tile_workspace_m = std::min(
      static_cast<size_t>(logical_m),
      static_cast<size_t>(read_env_uint("GROUP_QUANT_SPLIT_M", 32U)));
  const size_t tile_x_workspace =
      static_cast<size_t>(block_dim) * static_cast<size_t>(k_group_num) *
      tile_workspace_m *
      static_cast<size_t>(k_group_size) * sizeof(int8_t);
  // Tile-local scale workspace is compact [core, group, tile_m]. The kernel
  // writes each scalar scale through UB + MTE, then reads it back with MTE.
  const size_t tile_scale_workspace =
      static_cast<size_t>(block_dim) * static_cast<size_t>(k_group_num) *
      tile_workspace_m * sizeof(float);
  const size_t tile_x_workspace_offset =
      align_up(user_workspace + tile_acc_workspace, 512);
  const size_t tile_scale_workspace_offset =
      align_up(tile_x_workspace_offset + tile_x_workspace, 512);
  const size_t tile_workspace_end =
      align_up(tile_scale_workspace_offset + tile_scale_workspace, 512);
  const size_t workspace_size =
      tile_workspace_end + acc_workspace + system_workspace;
  auto workspace = at::empty(
      {static_cast<long>(workspace_size)}, x.options().dtype(at::kByte));

  auto stream = c10_npu::getCurrentNPUStream().stream();
  auto* bias_ptr = static_cast<void*>(nullptr);
  uint32_t ret = aclrtlaunch_quant_group_matmul_tile_dynamic_custom(
      static_cast<uint32_t>(block_dim),
      stream,
      const_cast<void*>(x.data_ptr()),
      const_cast<void*>(weight_nz_int8.data_ptr()),
      bias_ptr,
      cached.group_list.data_ptr(),
      const_cast<void*>(scale_f32.data_ptr()),
      x_q.data_ptr(),
      pertoken_scale.data_ptr(),
      y.data_ptr(),
      workspace.data_ptr(),
      cached.tiling_npu.data_ptr());
  TORCH_CHECK(ret == 0, "aclrtlaunch_quant_group_matmul_tile_dynamic_custom failed, ret=", ret);
  return y;
}

at::Tensor quant_group_matmul_forward_k_grouped_dynamic_external_2kernel(
    const at::Tensor& x,
    const at::Tensor& weight_nz_int8,
    const at::Tensor& scale_f32,
    int64_t k_group_size,
    int64_t block_dim = 8,
    int64_t parall_num = 4) {
  check_npu(x, "x");
  check_npu(weight_nz_int8, "weight_nz_int8");
  check_npu(scale_f32, "scale_f32");
  TORCH_CHECK(x.dim() == 2, "x must be 2D [M, K].");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "external 2-kernel dynamic prototype supports bf16 input only.");
  TORCH_CHECK(weight_nz_int8.scalar_type() == at::kChar, "weight_nz_int8 must be int8.");
  TORCH_CHECK(scale_f32.scalar_type() == at::kFloat, "scale_f32 must be float32.");
  TORCH_CHECK(scale_f32.dim() == 2, "scale_f32 must be 2D [k_group_num, N].");
  TORCH_CHECK(k_group_size > 0, "k_group_size must be > 0.");
  TORCH_CHECK(x.size(1) % k_group_size == 0, "K must be divisible by k_group_size.");

  const int64_t logical_m = x.size(0);
  const int64_t k_group_num = x.size(1) / k_group_size;
  const int64_t n = scale_f32.size(1);
  TORCH_CHECK(scale_f32.size(0) == k_group_num, "scale_f32 first dim must equal K group count.");

  auto cached = get_cached_launch_tensors(
      x.device().index(),
      logical_m,
      k_group_size,
      n,
      k_group_num,
      1,
      k_group_size,
      k_group_num,
      block_dim,
      parall_num);
  auto x_q = at::empty(
      {logical_m * k_group_num, k_group_size},
      x.options().dtype(at::kChar));
  auto pertoken_scale = at::empty(
      {logical_m * k_group_num},
      x.options().dtype(at::kFloat));
  auto y = at::empty({logical_m, n}, x.options().dtype(at::kHalf));

  auto stream = c10_npu::getCurrentNPUStream().stream();
  auto dynamic_workspace = at::empty({1}, x.options().dtype(at::kByte));
  uint32_t ret = aclrtlaunch_quant_group_dynamic_pack_custom(
      static_cast<uint32_t>(block_dim),
      stream,
      const_cast<void*>(x.data_ptr()),
      x_q.data_ptr(),
      pertoken_scale.data_ptr(),
      dynamic_workspace.data_ptr(),
      cached.tiling_npu.data_ptr());
  TORCH_CHECK(ret == 0, "aclrtlaunch_quant_group_dynamic_pack_custom failed, ret=", ret);

  auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
  const size_t system_workspace =
      static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
  const size_t user_workspace =
      static_cast<size_t>(parall_num) * 256ULL * 128ULL * sizeof(int32_t) *
      static_cast<size_t>(block_dim) * static_cast<size_t>(k_group_num);
  const bool keep_acc_workspace =
      std::getenv("GROUP_QUANT_KEEP_ACC_WORKSPACE") != nullptr;
  const size_t acc_workspace =
      keep_acc_workspace
          ? static_cast<size_t>(logical_m) * static_cast<size_t>(n) * sizeof(float)
          : 0;
  const size_t workspace_size = user_workspace + acc_workspace + system_workspace;
  auto workspace = at::empty(
      {static_cast<long>(workspace_size)}, x.options().dtype(at::kByte));

  auto* bias_ptr = static_cast<void*>(nullptr);
  ret = aclrtlaunch_quant_group_matmul_custom(
      static_cast<uint32_t>(block_dim),
      stream,
      x_q.data_ptr(),
      const_cast<void*>(weight_nz_int8.data_ptr()),
      bias_ptr,
      cached.group_list.data_ptr(),
      const_cast<void*>(scale_f32.data_ptr()),
      pertoken_scale.data_ptr(),
      y.data_ptr(),
      workspace.data_ptr(),
      cached.tiling_npu.data_ptr());
  TORCH_CHECK(ret == 0, "aclrtlaunch_quant_group_matmul_custom failed, ret=", ret);
  return y;
}

at::Tensor debug_quant_group_matmul_forward_k_grouped_passthrough_fused(
    const at::Tensor& x_int8,
    const at::Tensor& weight_nz_int8,
    const at::Tensor& scale_f32,
    const at::Tensor& pertoken_scale_f32,
    int64_t k_group_size,
    int64_t k_group_num,
    int64_t block_dim = 8,
    int64_t parall_num = 4) {
  TORCH_CHECK(x_int8.scalar_type() == at::kChar, "x_int8 must be int8.");
  TORCH_CHECK(weight_nz_int8.scalar_type() == at::kChar, "weight_nz_int8 must be int8.");
  TORCH_CHECK(scale_f32.scalar_type() == at::kFloat, "scale_f32 must be float32.");
  TORCH_CHECK(pertoken_scale_f32.scalar_type() == at::kFloat, "pertoken_scale_f32 must be float32.");
  TORCH_CHECK(x_int8.dim() == 2, "x_int8 must be 2D.");
  TORCH_CHECK(scale_f32.dim() == 2, "scale_f32 must be 2D [k_group_num, N].");
  TORCH_CHECK(pertoken_scale_f32.dim() == 1, "pertoken_scale_f32 must be 1D.");
  check_npu(x_int8, "x_int8");
  check_npu(weight_nz_int8, "weight_nz_int8");
  check_npu(scale_f32, "scale_f32");
  check_npu(pertoken_scale_f32, "pertoken_scale_f32");

  TORCH_CHECK(k_group_size > 0, "k_group_size must be > 0.");
  TORCH_CHECK(k_group_num > 0, "k_group_num must be > 0.");
  const int64_t x_m = x_int8.size(0);
  const int64_t x_k = x_int8.size(1);
  const int64_t n = scale_f32.size(1);
  TORCH_CHECK(x_k == k_group_size, "x_int8 second dim must equal k_group_size.");
  TORCH_CHECK(x_m % k_group_num == 0, "x_int8 first dim must be divisible by k_group_num.");
  TORCH_CHECK(scale_f32.size(0) == k_group_num, "scale_f32 first dim must equal k_group_num.");
  TORCH_CHECK(pertoken_scale_f32.size(0) == x_m, "pertoken_scale_f32 size must equal x_int8 first dim.");
  const int64_t logical_m = x_m / k_group_num;

  auto cached = get_cached_launch_tensors(
      x_int8.device().index(),
      logical_m,
      k_group_size,
      n,
      k_group_num,
      1,
      k_group_size,
      k_group_num,
      block_dim,
      parall_num);
  auto y = at::empty({logical_m, n}, x_int8.options().dtype(at::kHalf));

  auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
  const size_t system_workspace =
      static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
  const size_t x_q_workspace =
      static_cast<size_t>(logical_m) * static_cast<size_t>(k_group_num) *
      static_cast<size_t>(k_group_size) * sizeof(int8_t);
  const size_t pertoken_workspace =
      static_cast<size_t>(logical_m) * static_cast<size_t>(k_group_num) * sizeof(float);
  const size_t x_q_workspace_offset = align_up(kDynamicWorkspaceGuardBytes, 512);
  const size_t pertoken_workspace_offset =
      align_up(x_q_workspace_offset + align_up(x_q_workspace, 32), 512);
  const size_t matmul_workspace_offset =
      align_up(pertoken_workspace_offset + align_up(pertoken_workspace, 32), 512);
  const size_t user_workspace =
      static_cast<size_t>(parall_num) * 256ULL * 128ULL * sizeof(int32_t) *
      static_cast<size_t>(block_dim) * static_cast<size_t>(k_group_num);
  const bool keep_acc_workspace =
      std::getenv("GROUP_QUANT_KEEP_ACC_WORKSPACE") != nullptr;
  const size_t acc_workspace =
      keep_acc_workspace
          ? static_cast<size_t>(logical_m) * static_cast<size_t>(n) * sizeof(float)
          : 0;
  const size_t workspace_size =
      matmul_workspace_offset + user_workspace + acc_workspace + system_workspace;
  auto workspace = at::empty(
      {static_cast<long>(workspace_size)}, x_int8.options().dtype(at::kByte));

  auto stream = c10_npu::getCurrentNPUStream().stream();
  auto* bias_ptr = static_cast<void*>(nullptr);
  uint32_t ret = aclrtlaunch_quant_group_matmul_passthrough_dynamic_custom(
      static_cast<uint32_t>(block_dim),
      stream,
      const_cast<void*>(x_int8.data_ptr()),
      const_cast<void*>(weight_nz_int8.data_ptr()),
      bias_ptr,
      cached.group_list.data_ptr(),
      const_cast<void*>(scale_f32.data_ptr()),
      const_cast<void*>(pertoken_scale_f32.data_ptr()),
      y.data_ptr(),
      workspace.data_ptr(),
      cached.tiling_npu.data_ptr());
  TORCH_CHECK(ret == 0, "aclrtlaunch_quant_group_matmul_passthrough_dynamic_custom failed, ret=", ret);
  return y;
}

at::Tensor debug_quant_group_matmul_forward_k_grouped_workspace_passthrough_fused(
    const at::Tensor& x_int8,
    const at::Tensor& weight_nz_int8,
    const at::Tensor& scale_f32,
    const at::Tensor& pertoken_scale_f32,
    int64_t k_group_size,
    int64_t k_group_num,
    int64_t block_dim = 8,
    int64_t parall_num = 4) {
  TORCH_CHECK(x_int8.scalar_type() == at::kChar, "x_int8 must be int8.");
  TORCH_CHECK(weight_nz_int8.scalar_type() == at::kChar, "weight_nz_int8 must be int8.");
  TORCH_CHECK(scale_f32.scalar_type() == at::kFloat, "scale_f32 must be float32.");
  TORCH_CHECK(pertoken_scale_f32.scalar_type() == at::kFloat, "pertoken_scale_f32 must be float32.");
  TORCH_CHECK(x_int8.dim() == 2, "x_int8 must be 2D.");
  TORCH_CHECK(scale_f32.dim() == 2, "scale_f32 must be 2D [k_group_num, N].");
  TORCH_CHECK(pertoken_scale_f32.dim() == 1, "pertoken_scale_f32 must be 1D.");
  check_npu(x_int8, "x_int8");
  check_npu(weight_nz_int8, "weight_nz_int8");
  check_npu(scale_f32, "scale_f32");
  check_npu(pertoken_scale_f32, "pertoken_scale_f32");

  TORCH_CHECK(k_group_size > 0, "k_group_size must be > 0.");
  TORCH_CHECK(k_group_num > 0, "k_group_num must be > 0.");
  const int64_t x_m = x_int8.size(0);
  const int64_t x_k = x_int8.size(1);
  const int64_t n = scale_f32.size(1);
  TORCH_CHECK(x_k == k_group_size, "x_int8 second dim must equal k_group_size.");
  TORCH_CHECK(x_m % k_group_num == 0, "x_int8 first dim must be divisible by k_group_num.");
  TORCH_CHECK(scale_f32.size(0) == k_group_num, "scale_f32 first dim must equal k_group_num.");
  TORCH_CHECK(pertoken_scale_f32.size(0) == x_m, "pertoken_scale_f32 size must equal x_int8 first dim.");
  const int64_t logical_m = x_m / k_group_num;

  auto cached = get_cached_launch_tensors(
      x_int8.device().index(),
      logical_m,
      k_group_size,
      n,
      k_group_num,
      1,
      k_group_size,
      k_group_num,
      block_dim,
      parall_num);
  auto y = at::empty({logical_m, n}, x_int8.options().dtype(at::kHalf));

  auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
  const size_t system_workspace =
      static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
  const size_t x_q_workspace =
      static_cast<size_t>(logical_m) * static_cast<size_t>(k_group_num) *
      static_cast<size_t>(k_group_size) * sizeof(int8_t);
  const size_t pertoken_workspace =
      static_cast<size_t>(logical_m) * static_cast<size_t>(k_group_num) * sizeof(float);
  const size_t x_q_workspace_offset = align_up(kDynamicWorkspaceGuardBytes, 512);
  const size_t pertoken_workspace_offset =
      align_up(x_q_workspace_offset + align_up(x_q_workspace, 32), 512);
  const size_t matmul_workspace_offset =
      align_up(pertoken_workspace_offset + align_up(pertoken_workspace, 32), 512);
  const size_t user_workspace =
      static_cast<size_t>(parall_num) * 256ULL * 128ULL * sizeof(int32_t) *
      static_cast<size_t>(block_dim) * static_cast<size_t>(k_group_num);
  const bool keep_acc_workspace =
      std::getenv("GROUP_QUANT_KEEP_ACC_WORKSPACE") != nullptr;
  const size_t acc_workspace =
      keep_acc_workspace
          ? static_cast<size_t>(logical_m) * static_cast<size_t>(n) * sizeof(float)
          : 0;
  const size_t workspace_size =
      matmul_workspace_offset + user_workspace + acc_workspace + system_workspace;
  auto workspace = at::empty(
      {static_cast<long>(workspace_size)}, x_int8.options().dtype(at::kByte));

  auto stream = c10_npu::getCurrentNPUStream().stream();
  auto* bias_ptr = static_cast<void*>(nullptr);
  uint32_t ret = aclrtlaunch_quant_group_matmul_workspace_passthrough_dynamic_custom(
      static_cast<uint32_t>(block_dim),
      stream,
      const_cast<void*>(x_int8.data_ptr()),
      const_cast<void*>(weight_nz_int8.data_ptr()),
      bias_ptr,
      cached.group_list.data_ptr(),
      const_cast<void*>(scale_f32.data_ptr()),
      const_cast<void*>(pertoken_scale_f32.data_ptr()),
      y.data_ptr(),
      workspace.data_ptr(),
      cached.tiling_npu.data_ptr());
  TORCH_CHECK(ret == 0, "aclrtlaunch_quant_group_matmul_workspace_passthrough_dynamic_custom failed, ret=", ret);
  return y;
}

std::tuple<at::Tensor, at::Tensor> debug_dynamic_pack_from_matmul_kernel(
    const at::Tensor& x,
    int64_t k_group_size,
    int64_t block_dim = 8,
    int64_t parall_num = 4) {
  check_npu(x, "x");
  TORCH_CHECK(x.dim() == 2, "x must be 2D [M, K].");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "debug dynamic pack supports bf16 input only.");
  TORCH_CHECK(k_group_size > 0, "k_group_size must be > 0.");
  TORCH_CHECK(x.size(1) % k_group_size == 0, "K must be divisible by k_group_size.");

  const int64_t logical_m = x.size(0);
  const int64_t k_group_num = x.size(1) / k_group_size;
  auto cached = get_cached_launch_tensors(
      x.device().index(),
      logical_m,
      k_group_size,
      /*n=*/k_group_size,
      k_group_num,
      1,
      k_group_size,
      k_group_num,
      block_dim,
      parall_num);
  auto x_q = at::empty({logical_m * k_group_num, k_group_size}, x.options().dtype(at::kChar));
  auto pertoken_scale = at::empty({logical_m * k_group_num}, x.options().dtype(at::kFloat));
  auto workspace = at::empty({1}, x.options().dtype(at::kByte));

  auto stream = c10_npu::getCurrentNPUStream().stream();
  uint32_t ret = aclrtlaunch_quant_group_dynamic_pack_custom(
      static_cast<uint32_t>(block_dim),
      stream,
      const_cast<void*>(x.data_ptr()),
      x_q.data_ptr(),
      pertoken_scale.data_ptr(),
      workspace.data_ptr(),
      cached.tiling_npu.data_ptr());
  TORCH_CHECK(ret == 0, "aclrtlaunch_quant_group_dynamic_pack_custom failed, ret=", ret);
  return {x_q, pertoken_scale};
}

std::tuple<at::Tensor, at::Tensor> debug_dynamic_pack_aic_read_from_matmul_kernel(
    const at::Tensor& x,
    int64_t k_group_size,
    int64_t block_dim = 8,
    int64_t parall_num = 4) {
  check_npu(x, "x");
  TORCH_CHECK(x.dim() == 2, "x must be 2D [M, K].");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "debug dynamic pack supports bf16 input only.");
  TORCH_CHECK(k_group_size > 0, "k_group_size must be > 0.");
  TORCH_CHECK(x.size(1) % k_group_size == 0, "K must be divisible by k_group_size.");

  const int64_t logical_m = x.size(0);
  const int64_t k_group_num = x.size(1) / k_group_size;
  auto cached = get_cached_launch_tensors(
      x.device().index(),
      logical_m,
      k_group_size,
      /*n=*/k_group_size,
      k_group_num,
      1,
      k_group_size,
      k_group_num,
      block_dim,
      parall_num);
  auto x_q = at::empty({logical_m * k_group_num, k_group_size}, x.options().dtype(at::kChar));
  auto pertoken_scale = at::empty({logical_m * k_group_num}, x.options().dtype(at::kFloat));

  const size_t x_q_workspace =
      static_cast<size_t>(logical_m) * static_cast<size_t>(k_group_num) *
      static_cast<size_t>(k_group_size) * sizeof(int8_t);
  const size_t pertoken_workspace =
      static_cast<size_t>(logical_m) * static_cast<size_t>(k_group_num) * sizeof(float);
  const size_t x_q_workspace_offset = align_up(kDynamicWorkspaceGuardBytes, 512);
  const size_t pertoken_workspace_offset =
      align_up(x_q_workspace_offset + align_up(x_q_workspace, 32), 512);
  const size_t workspace_size =
      pertoken_workspace_offset + align_up(pertoken_workspace, 32);
  auto workspace = at::empty({static_cast<long>(workspace_size)}, x.options().dtype(at::kByte));

  auto stream = c10_npu::getCurrentNPUStream().stream();
  uint32_t ret = aclrtlaunch_quant_group_dynamic_pack_aic_read_custom(
      static_cast<uint32_t>(block_dim),
      stream,
      const_cast<void*>(x.data_ptr()),
      x_q.data_ptr(),
      pertoken_scale.data_ptr(),
      workspace.data_ptr(),
      cached.tiling_npu.data_ptr());
  TORCH_CHECK(ret == 0, "aclrtlaunch_quant_group_dynamic_pack_aic_read_custom failed, ret=", ret);
  return {x_q, pertoken_scale};
}

std::tuple<at::Tensor, at::Tensor> debug_tile_workspace_roundtrip(
    const at::Tensor& x_q,
    const at::Tensor& pertoken_scale,
    int64_t k_group_size,
    int64_t k_group_num,
    int64_t block_dim = 8,
    int64_t parall_num = 4) {
  check_npu(x_q, "x_q");
  check_npu(pertoken_scale, "pertoken_scale");
  TORCH_CHECK(x_q.dim() == 2, "x_q must be 2D [M * k_group_num, k_group_size].");
  TORCH_CHECK(x_q.scalar_type() == at::kChar, "x_q must be int8.");
  TORCH_CHECK(pertoken_scale.dim() == 1, "pertoken_scale must be 1D [M * k_group_num].");
  TORCH_CHECK(pertoken_scale.scalar_type() == at::kFloat, "pertoken_scale must be float32.");
  TORCH_CHECK(k_group_size > 0, "k_group_size must be > 0.");
  TORCH_CHECK(k_group_num > 0, "k_group_num must be > 0.");
  TORCH_CHECK(x_q.size(1) == k_group_size, "x_q second dim must equal k_group_size.");
  TORCH_CHECK(x_q.size(0) % k_group_num == 0, "x_q first dim must be divisible by k_group_num.");
  TORCH_CHECK(pertoken_scale.size(0) == x_q.size(0), "pertoken_scale size must match x_q rows.");

  const int64_t logical_m = x_q.size(0) / k_group_num;
  auto cached = get_cached_launch_tensors(
      x_q.device().index(),
      logical_m,
      k_group_size,
      /*n=*/k_group_size,
      k_group_num,
      1,
      k_group_size,
      k_group_num,
      block_dim,
      parall_num);

  auto x_q_out = at::empty_like(x_q);
  auto pertoken_scale_out = at::empty_like(pertoken_scale);
  const size_t tile_x_workspace =
      static_cast<size_t>(k_group_num) * static_cast<size_t>(logical_m) *
      static_cast<size_t>(k_group_size) * sizeof(int8_t);
  const size_t tile_scale_workspace =
      static_cast<size_t>(k_group_num) * static_cast<size_t>(logical_m) * sizeof(float);
  const size_t tile_scale_offset = align_up(tile_x_workspace, 512);
  const size_t workspace_size = tile_scale_offset + align_up(tile_scale_workspace, 512);
  auto workspace = at::empty({static_cast<long>(workspace_size)}, x_q.options().dtype(at::kByte));

  auto stream = c10_npu::getCurrentNPUStream().stream();
  uint32_t ret = aclrtlaunch_quant_group_tile_workspace_roundtrip_custom(
      static_cast<uint32_t>(block_dim),
      stream,
      const_cast<void*>(x_q.data_ptr()),
      const_cast<void*>(pertoken_scale.data_ptr()),
      x_q_out.data_ptr(),
      pertoken_scale_out.data_ptr(),
      workspace.data_ptr(),
      cached.tiling_npu.data_ptr());
  TORCH_CHECK(ret == 0, "aclrtlaunch_quant_group_tile_workspace_roundtrip_custom failed, ret=", ret);
  return {x_q_out, pertoken_scale_out};
}

std::tuple<at::Tensor, at::Tensor> debug_tile_matmul_compare(
    const at::Tensor& x_q,
    const at::Tensor& weight_nz_int8,
    const at::Tensor& scale_f32,
    int64_t k_group_size,
    int64_t k_group_num,
    int64_t parall_num = 4) {
  check_npu(x_q, "x_q");
  check_npu(weight_nz_int8, "weight_nz_int8");
  check_npu(scale_f32, "scale_f32");
  TORCH_CHECK(x_q.dim() == 2, "x_q must be 2D [M * k_group_num, k_group_size].");
  TORCH_CHECK(x_q.scalar_type() == at::kChar, "x_q must be int8.");
  TORCH_CHECK(weight_nz_int8.scalar_type() == at::kChar, "weight_nz_int8 must be int8.");
  TORCH_CHECK(scale_f32.dim() == 2, "scale_f32 must be 2D [k_group_num, N].");
  TORCH_CHECK(scale_f32.scalar_type() == at::kFloat, "scale_f32 must be float32.");
  TORCH_CHECK(k_group_size > 0, "k_group_size must be > 0.");
  TORCH_CHECK(k_group_num > 0, "k_group_num must be > 0.");
  TORCH_CHECK(x_q.size(1) == k_group_size, "x_q second dim must equal k_group_size.");
  TORCH_CHECK(x_q.size(0) % k_group_num == 0, "x_q first dim must be divisible by k_group_num.");
  TORCH_CHECK(scale_f32.size(0) == k_group_num, "scale first dim must equal k_group_num.");

  const int64_t logical_m = x_q.size(0) / k_group_num;
  const int64_t n = scale_f32.size(1);
  const int64_t block_dim = 1;
  auto cached = get_cached_launch_tensors(
      x_q.device().index(),
      logical_m,
      k_group_size,
      n,
      k_group_num,
      1,
      k_group_size,
      k_group_num,
      block_dim,
      parall_num);

  auto full_out = at::empty({logical_m, n}, x_q.options().dtype(at::kInt));
  auto tile_out = at::empty({logical_m, n}, x_q.options().dtype(at::kInt));
  const size_t tile_x_workspace =
      static_cast<size_t>(logical_m) * static_cast<size_t>(k_group_size) * sizeof(int8_t);
  auto workspace = at::empty({static_cast<long>(align_up(tile_x_workspace, 512))},
                             x_q.options().dtype(at::kByte));

  auto stream = c10_npu::getCurrentNPUStream().stream();
  uint32_t ret = aclrtlaunch_quant_group_tile_matmul_compare_custom(
      static_cast<uint32_t>(block_dim),
      stream,
      const_cast<void*>(x_q.data_ptr()),
      const_cast<void*>(weight_nz_int8.data_ptr()),
      full_out.data_ptr(),
      tile_out.data_ptr(),
      workspace.data_ptr(),
      cached.tiling_npu.data_ptr());
  TORCH_CHECK(ret == 0, "aclrtlaunch_quant_group_tile_matmul_compare_custom failed, ret=", ret);
  return {full_out, tile_out};
}

std::tuple<at::Tensor, at::Tensor> debug_shared_tile_pack(
    const at::Tensor& x,
    int64_t k_group_size,
    int64_t block_dim = 8,
    int64_t parall_num = 4) {
  check_npu(x, "x");
  TORCH_CHECK(x.dim() == 2, "x must be 2D [M, K].");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "debug shared tile pack supports bf16 input only.");
  TORCH_CHECK(k_group_size > 0, "k_group_size must be > 0.");
  TORCH_CHECK(x.size(1) % k_group_size == 0, "K must be divisible by k_group_size.");

  const int64_t logical_m = x.size(0);
  const int64_t k_group_num = x.size(1) / k_group_size;
  auto cached = get_cached_launch_tensors(
      x.device().index(),
      logical_m,
      k_group_size,
      /*n=*/k_group_size,
      k_group_num,
      1,
      k_group_size,
      k_group_num,
      block_dim,
      parall_num);

  auto x_q = at::empty({logical_m * k_group_num, k_group_size}, x.options().dtype(at::kChar));
  auto pertoken_scale = at::empty({logical_m * k_group_num}, x.options().dtype(at::kFloat));
  // The debug kernel addresses its compact tile workspace with
  // mmTilingData.baseM, which may be larger than logical_m for small shapes.
  // Keep this allocation deliberately generous so the debug tool does not
  // introduce its own OOB writes.
  const int64_t debug_tile_m = logical_m > 256 ? logical_m : 256;
  const size_t tile_x_workspace =
      static_cast<size_t>(k_group_num) *
      static_cast<size_t>(debug_tile_m) *
      static_cast<size_t>(k_group_size) * sizeof(int8_t);
  const size_t tile_scale_workspace =
      static_cast<size_t>(k_group_num) * static_cast<size_t>(debug_tile_m) * sizeof(float);
  const size_t tile_scale_offset = align_up(tile_x_workspace, 512);
  const size_t workspace_size = tile_scale_offset + align_up(tile_scale_workspace, 512);
  auto workspace = at::empty({static_cast<long>(workspace_size)}, x.options().dtype(at::kByte));

  auto stream = c10_npu::getCurrentNPUStream().stream();
  uint32_t ret = aclrtlaunch_quant_group_shared_tile_pack_debug_custom(
      static_cast<uint32_t>(block_dim),
      stream,
      const_cast<void*>(x.data_ptr()),
      x_q.data_ptr(),
      pertoken_scale.data_ptr(),
      workspace.data_ptr(),
      cached.tiling_npu.data_ptr());
  TORCH_CHECK(ret == 0, "aclrtlaunch_quant_group_shared_tile_pack_debug_custom failed, ret=", ret);
  return {x_q, pertoken_scale};
}

std::tuple<at::Tensor, at::Tensor> debug_tile_dynamic_pack(
    const at::Tensor& x,
    int64_t k_group_size,
    int64_t block_dim = 8,
    int64_t parall_num = 4) {
  check_npu(x, "x");
  TORCH_CHECK(x.dim() == 2, "x must be 2D [M, K].");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "debug tile dynamic pack supports bf16 input only.");
  TORCH_CHECK(k_group_size > 0, "k_group_size must be > 0.");
  TORCH_CHECK(x.size(1) % k_group_size == 0, "K must be divisible by k_group_size.");

  const int64_t logical_m = x.size(0);
  const int64_t k_group_num = x.size(1) / k_group_size;
  auto cached = get_cached_launch_tensors(
      x.device().index(),
      logical_m,
      k_group_size,
      /*n=*/k_group_size,
      k_group_num,
      1,
      k_group_size,
      k_group_num,
      block_dim,
      parall_num);

  auto x_q = at::empty({logical_m * k_group_num, k_group_size}, x.options().dtype(at::kChar));
  auto pertoken_scale = at::empty({logical_m * k_group_num}, x.options().dtype(at::kFloat));
  auto workspace = at::empty({1}, x.options().dtype(at::kByte));

  auto stream = c10_npu::getCurrentNPUStream().stream();
  uint32_t ret = aclrtlaunch_quant_group_tile_dynamic_pack_debug_custom(
      static_cast<uint32_t>(block_dim),
      stream,
      const_cast<void*>(x.data_ptr()),
      x_q.data_ptr(),
      pertoken_scale.data_ptr(),
      workspace.data_ptr(),
      cached.tiling_npu.data_ptr());
  TORCH_CHECK(ret == 0, "aclrtlaunch_quant_group_tile_dynamic_pack_debug_custom failed, ret=", ret);
  return {x_q, pertoken_scale};
}

at::Tensor quant_group_matmul_forward(
    const at::Tensor& x_int8,
    const at::Tensor& weight_nz_int8,
    const at::Tensor& scale_f32,
    const at::Tensor& pertoken_scale_f32,
    int64_t block_dim = 8,
    int64_t parall_num = 4) {
  auto group_list = at::full(
      {1},
      x_int8.size(0),
      x_int8.options().dtype(at::kLong));
  auto grouped_scale =
      scale_f32.dim() == 1 ? scale_f32.unsqueeze(0) : scale_f32;
  return quant_group_matmul_forward_grouped(
      x_int8,
      weight_nz_int8,
      grouped_scale,
      pertoken_scale_f32,
      group_list,
      0,
      0,
      0,
      block_dim,
      parall_num);
}

at::Tensor quant_group_matmul_forward_int8_grouped(
    const at::Tensor& x_int8,
    const at::Tensor& weight_nz_int8,
    const at::Tensor& scale_f32,
    const at::Tensor& pertoken_scale_f32,
    const at::Tensor& group_list,
    int64_t group_axis = 0,
    int64_t k_group_size = 0,
    int64_t k_group_num = 0,
    const at::Tensor& bias_int8 = at::Tensor(),
    int64_t block_dim = 8,
    int64_t parall_num = 4) {
  check_common_inputs(
      x_int8,
      weight_nz_int8,
      scale_f32,
      pertoken_scale_f32,
      group_list);

  const int64_t x_m = x_int8.size(0);
  const int64_t x_k = x_int8.size(1);
  const int64_t n = scale_f32.size(1);
  TORCH_CHECK(group_axis == 0 || group_axis == 1 || group_axis == 2, "group_axis must be 0, 1 or 2.");
  int64_t logical_group_num = group_list.size(0);
  const int64_t logical_m = group_list.sum().item<int64_t>();
  if (group_axis == 0) {
    TORCH_CHECK(pertoken_scale_f32.size(0) == x_m, "pertoken_scale_f32 size mismatch with M.");
    TORCH_CHECK(group_list.sum().item<int64_t>() == x_m, "sum(group_list) must equal M.");
    TORCH_CHECK(
        scale_f32.size(0) == logical_group_num,
        "scale_f32 group dimension mismatch for M-axis grouping.");
  }
  if (group_axis == 1) {
    TORCH_CHECK(k_group_size > 0, "k_group_size must be > 0 for K-axis grouping.");
    TORCH_CHECK(k_group_num > 0, "k_group_num must be > 0 for K-axis grouping.");
    TORCH_CHECK(
        group_list.size(0) == 1,
        "group_list must have a single entry [M] for K-axis grouping.");
    TORCH_CHECK(
        scale_f32.size(0) == k_group_num,
        "scale_f32 must have shape [k_group_num, N] for K-axis grouping.");
    TORCH_CHECK(
        x_k == k_group_size,
        "x_int8 second dim must equal k_group_size for K-axis grouping.");
    TORCH_CHECK(
        x_m == logical_m * k_group_num,
        "x_int8 first dim must equal M * k_group_num for K-axis grouping.");
    TORCH_CHECK(
        pertoken_scale_f32.size(0) == logical_m * k_group_num,
        "pertoken_scale_f32 must have shape [k_group_num * M] for K-axis grouping.");
    TORCH_CHECK(
        false,
        "K-axis grouped INT8 output path is not enabled yet; "
        "only fp16 host-loop accumulation is currently supported.");
  }
  if (group_axis == 2) {
    TORCH_CHECK(
        false,
        "Expert x K-axis grouped INT8 output path is not enabled yet; "
        "only fp16 output is currently supported.");
  }

  auto cached = get_cached_launch_tensors(
      x_int8.device().index(),
      logical_m,
      x_k,
      n,
      logical_group_num,
      group_axis,
      k_group_size,
      k_group_num,
      block_dim,
      parall_num);
  auto y = at::empty({logical_m, n}, x_int8.options().dtype(at::kChar));

  auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
  const size_t system_workspace =
      static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
  const size_t user_workspace =
      static_cast<size_t>(parall_num) * 256ULL * 128ULL * sizeof(int32_t) *
      static_cast<size_t>(block_dim);
  const size_t workspace_size = user_workspace + system_workspace;
  auto workspace = at::empty(
      {static_cast<long>(workspace_size)}, x_int8.options().dtype(at::kByte));

  auto stream = c10_npu::getCurrentNPUStream().stream();
  auto* bias_ptr = static_cast<void*>(nullptr);  // kernel-side bias is unused in this path
  uint32_t ret = aclrtlaunch_quant_group_matmul_custom_int8(
      static_cast<uint32_t>(block_dim),
      stream,
      const_cast<void*>(x_int8.data_ptr()),
      const_cast<void*>(weight_nz_int8.data_ptr()),
      bias_ptr,
      const_cast<void*>(group_list.data_ptr()),
      const_cast<void*>(scale_f32.data_ptr()),
      const_cast<void*>(pertoken_scale_f32.data_ptr()),
      y.data_ptr(),
      workspace.data_ptr(),
      cached.tiling_npu.data_ptr());
  TORCH_CHECK(ret == 0, "aclrtlaunch_quant_group_matmul_custom_int8 failed, ret=", ret);

  if (bias_int8.defined() && bias_int8.numel() > 0) {
    TORCH_CHECK(bias_int8.scalar_type() == at::kChar, "bias_int8 must be int8.");
    TORCH_CHECK(bias_int8.dim() == 1, "bias_int8 must be 1D [N].");
    TORCH_CHECK(bias_int8.size(0) == n, "bias_int8 size mismatch with N.");
    check_npu(bias_int8, "bias_int8");
    auto y_i16 = y.to(at::kShort);
    y_i16.add_(bias_int8.view({1, n}).to(at::kShort));
    y = y_i16.clamp(-128, 127).to(at::kChar);
  }
  return y;
}

at::Tensor quant_group_matmul_forward_int8(
    const at::Tensor& x_int8,
    const at::Tensor& weight_nz_int8,
    const at::Tensor& scale_f32,
    const at::Tensor& pertoken_scale_f32,
    const at::Tensor& bias_int8 = at::Tensor(),
    int64_t block_dim = 8,
    int64_t parall_num = 4) {
  auto group_list = at::full(
      {1},
      x_int8.size(0),
      x_int8.options().dtype(at::kLong));
  auto grouped_scale =
      scale_f32.dim() == 1 ? scale_f32.unsqueeze(0) : scale_f32;
  return quant_group_matmul_forward_int8_grouped(
      x_int8,
      weight_nz_int8,
      grouped_scale,
      pertoken_scale_f32,
      group_list,
      0,
      0,
      0,
      bias_int8,
      block_dim,
      parall_num);
}

TORCH_LIBRARY_FRAGMENT(vllm_smooth, ops) {
  ops.def("dynamic_quant_k_grouped_ascendc(Tensor x, int group_size, int block_dim=8) -> (Tensor, Tensor)");
  ops.def("quant_matmul_grouped(Tensor x, Tensor weight, Tensor weight_scale, Tensor x_scale, Tensor group_list, int group_axis=0, int k_group_size=0, int k_group_num=0, int block_dim=8, int parall_num=4) -> Tensor");
  ops.def("quant_matmul_k_grouped(Tensor x, Tensor weight, Tensor weight_scale, Tensor x_scale, int k_group_size, int k_group_num, int block_dim=8, int parall_num=4) -> Tensor");
  ops.def("quant_matmul_k_grouped_dynamic(Tensor x, Tensor weight, Tensor weight_scale, int k_group_size, int block_dim=8, int parall_num=4) -> Tensor");
}

TORCH_LIBRARY_IMPL(vllm_smooth, PrivateUse1, ops) {
  ops.impl("dynamic_quant_k_grouped_ascendc", TORCH_FN(dynamic_quant_k_grouped_ascendc));
  ops.impl("quant_matmul_grouped", TORCH_FN(quant_group_matmul_forward_grouped));
  ops.impl("quant_matmul_k_grouped", TORCH_FN(quant_group_matmul_forward_k_grouped));
  ops.impl("quant_matmul_k_grouped_dynamic", TORCH_FN(quant_group_matmul_forward_k_grouped_tile_dynamic));
}

namespace {
std::tuple<at::Tensor, at::Tensor> dynamic_quant_k_grouped_meta(
    const at::Tensor& x, int64_t group_size, int64_t) {
  TORCH_CHECK(x.dim() == 2 && group_size > 0 && x.size(1) % group_size == 0,
              "x must be [M, K] and K must be divisible by group_size.");
  const auto groups = x.size(1) / group_size;
  return {
      at::empty({groups * x.size(0), group_size}, x.options().dtype(at::kChar)),
      at::empty({groups, x.size(0)}, x.options().dtype(at::kFloat)),
  };
}

at::Tensor quant_matmul_grouped_meta(
    const at::Tensor& x, const at::Tensor&, const at::Tensor& weight_scale,
    const at::Tensor&, const at::Tensor&, int64_t group_axis,
    int64_t, int64_t k_group_num, int64_t, int64_t) {
  const auto m = group_axis == 0 ? x.size(0) : x.size(0) / k_group_num;
  return at::empty({m, weight_scale.size(1)}, x.options().dtype(at::kHalf));
}

at::Tensor quant_matmul_k_grouped_meta(
    const at::Tensor& x, const at::Tensor&, const at::Tensor& weight_scale,
    const at::Tensor&, int64_t, int64_t k_group_num, int64_t, int64_t) {
  return at::empty(
      {x.size(0) / k_group_num, weight_scale.size(1)},
      x.options().dtype(at::kHalf));
}

at::Tensor quant_matmul_k_grouped_dynamic_meta(
    const at::Tensor& x, const at::Tensor&, const at::Tensor& weight_scale,
    int64_t, int64_t, int64_t) {
  return at::empty({x.size(0), weight_scale.size(1)}, x.options().dtype(at::kHalf));
}
}  // namespace

TORCH_LIBRARY_IMPL(vllm_smooth, Meta, ops) {
  ops.impl("dynamic_quant_k_grouped_ascendc", TORCH_FN(dynamic_quant_k_grouped_meta));
  ops.impl("quant_matmul_grouped", TORCH_FN(quant_matmul_grouped_meta));
  ops.impl("quant_matmul_k_grouped", TORCH_FN(quant_matmul_k_grouped_meta));
  ops.impl("quant_matmul_k_grouped_dynamic", TORCH_FN(quant_matmul_k_grouped_dynamic_meta));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "pack_activation_k_grouped",
      &pack_activation_k_grouped,
      py::arg("x"),
      py::arg("group_size"));
  m.def(
      "dynamic_quant_k_grouped_ascendc",
      &dynamic_quant_k_grouped_ascendc,
      py::arg("x"),
      py::arg("group_size"),
      py::arg("block_dim") = 8);
  m.def(
      "forward",
      &quant_group_matmul_forward,
      py::arg("x_int8"),
      py::arg("weight_nz_int8"),
      py::arg("scale_f32"),
      py::arg("pertoken_scale_f32"),
      py::arg("block_dim") = 8,
      py::arg("parall_num") = 4);
  m.def(
      "forward_grouped",
      &quant_group_matmul_forward_grouped,
      py::arg("x_int8"),
      py::arg("weight_nz_int8"),
      py::arg("scale_f32"),
      py::arg("pertoken_scale_f32"),
      py::arg("group_list"),
      py::arg("group_axis") = 0,
      py::arg("k_group_size") = 0,
      py::arg("k_group_num") = 0,
      py::arg("block_dim") = 8,
      py::arg("parall_num") = 4);
  m.def(
      "forward_k_grouped",
      &quant_group_matmul_forward_k_grouped,
      py::arg("x_int8"),
      py::arg("weight_nz_int8"),
      py::arg("scale_f32"),
      py::arg("pertoken_scale_f32"),
      py::arg("k_group_size"),
      py::arg("k_group_num"),
      py::arg("block_dim") = 8,
      py::arg("parall_num") = 4);
  m.def(
      "forward_k_grouped_dynamic_fused",
      &quant_group_matmul_forward_k_grouped_dynamic_fused,
      py::arg("x"),
      py::arg("weight_nz_int8"),
      py::arg("scale_f32"),
      py::arg("k_group_size"),
      py::arg("block_dim") = 8,
      py::arg("parall_num") = 4);
  m.def(
      "forward_k_grouped_dynamic_external_fused",
      &quant_group_matmul_forward_k_grouped_dynamic_external_fused,
      py::arg("x"),
      py::arg("weight_nz_int8"),
      py::arg("scale_f32"),
      py::arg("k_group_size"),
      py::arg("block_dim") = 8,
      py::arg("parall_num") = 4);
  m.def(
      "forward_k_grouped_tile_dynamic",
      &quant_group_matmul_forward_k_grouped_tile_dynamic,
      py::arg("x"),
      py::arg("weight_nz_int8"),
      py::arg("scale_f32"),
      py::arg("k_group_size"),
      py::arg("block_dim") = 8,
      py::arg("parall_num") = 4);
  m.def(
      "forward_k_grouped_dynamic_external_2kernel",
      &quant_group_matmul_forward_k_grouped_dynamic_external_2kernel,
      py::arg("x"),
      py::arg("weight_nz_int8"),
      py::arg("scale_f32"),
      py::arg("k_group_size"),
      py::arg("block_dim") = 8,
      py::arg("parall_num") = 4);
  m.def(
      "debug_forward_k_grouped_passthrough_fused",
      &debug_quant_group_matmul_forward_k_grouped_passthrough_fused,
      py::arg("x_int8"),
      py::arg("weight_nz_int8"),
      py::arg("scale_f32"),
      py::arg("pertoken_scale_f32"),
      py::arg("k_group_size"),
      py::arg("k_group_num"),
      py::arg("block_dim") = 8,
      py::arg("parall_num") = 4);
  m.def(
      "debug_forward_k_grouped_workspace_passthrough_fused",
      &debug_quant_group_matmul_forward_k_grouped_workspace_passthrough_fused,
      py::arg("x_int8"),
      py::arg("weight_nz_int8"),
      py::arg("scale_f32"),
      py::arg("pertoken_scale_f32"),
      py::arg("k_group_size"),
      py::arg("k_group_num"),
      py::arg("block_dim") = 8,
      py::arg("parall_num") = 4);
  m.def(
      "debug_dynamic_pack_from_matmul_kernel",
      &debug_dynamic_pack_from_matmul_kernel,
      py::arg("x"),
      py::arg("k_group_size"),
      py::arg("block_dim") = 8,
      py::arg("parall_num") = 4);
  m.def(
      "debug_dynamic_pack_aic_read_from_matmul_kernel",
      &debug_dynamic_pack_aic_read_from_matmul_kernel,
      py::arg("x"),
      py::arg("k_group_size"),
      py::arg("block_dim") = 8,
      py::arg("parall_num") = 4);
  m.def(
      "debug_tile_workspace_roundtrip",
      &debug_tile_workspace_roundtrip,
      py::arg("x_q"),
      py::arg("pertoken_scale"),
      py::arg("k_group_size"),
      py::arg("k_group_num"),
      py::arg("block_dim") = 8,
      py::arg("parall_num") = 4);
  m.def(
      "debug_tile_matmul_compare",
      &debug_tile_matmul_compare,
      py::arg("x_q"),
      py::arg("weight_nz_int8"),
      py::arg("scale_f32"),
      py::arg("k_group_size"),
      py::arg("k_group_num"),
      py::arg("parall_num") = 4);
  m.def(
      "debug_shared_tile_pack",
      &debug_shared_tile_pack,
      py::arg("x"),
      py::arg("k_group_size"),
      py::arg("block_dim") = 8,
      py::arg("parall_num") = 4);
  m.def(
      "debug_tile_dynamic_pack",
      &debug_tile_dynamic_pack,
      py::arg("x"),
      py::arg("k_group_size"),
      py::arg("block_dim") = 8,
      py::arg("parall_num") = 4);
  m.def(
      "forward_int8",
      &quant_group_matmul_forward_int8,
      py::arg("x_int8"),
      py::arg("weight_nz_int8"),
      py::arg("scale_f32"),
      py::arg("pertoken_scale_f32"),
      py::arg("bias_int8") = at::Tensor(),
      py::arg("block_dim") = 8,
      py::arg("parall_num") = 4);
  m.def(
      "forward_int8_grouped",
      &quant_group_matmul_forward_int8_grouped,
      py::arg("x_int8"),
      py::arg("weight_nz_int8"),
      py::arg("scale_f32"),
      py::arg("pertoken_scale_f32"),
      py::arg("group_list"),
      py::arg("group_axis") = 0,
      py::arg("k_group_size") = 0,
      py::arg("k_group_num") = 0,
      py::arg("bias_int8") = at::Tensor(),
      py::arg("block_dim") = 8,
      py::arg("parall_num") = 4);
}

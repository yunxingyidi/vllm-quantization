## 目录结构介绍

```text
quant_group_matmul/
├── cmake
├── scripts
│   ├── gen_data.py
│   └── verify_result.py
├── CMakeLists.txt
├── data_utils.h
├── main.cpp
├── quant_group_matmul_custom_tiling.h
├── quant_group_matmul_custom_tiling.cpp
├── quant_group_matmul_custom.cpp
├── quant_group_matmul_tile_dynamic_custom.cpp
├── group_dynamic_quant_custom.cpp
├── python_api
└── run.sh
```

## 代码实现介绍

这一层包含了三部分：

1. host 直调样例
   `main.cpp` 构造输入、生成 tiling，并通过 `ACLRT_LAUNCH_KERNEL` 或
   `ICPU_RUN_KF` 直接调用 AscendC kernel。
2. 主 kernel
   `quant_group_matmul_custom.cpp` 是当前的基线实现。
3. 实验分支
   `quant_group_matmul_tile_dynamic_custom.cpp` 和
   `group_dynamic_quant_custom.cpp` 提供动态量化和 tile 级融合实验入口。

## 当前 kernel 语义

### 1. `group_axis=0`

沿 `M` 轴分组，语义与最初样例一致：

- `groupList[i]` 指定第 `i` 组 token 数
- `weight[i]` 是这一组对应的权重
- `scale[i]` 是这一组对应的 `[N]` 反量化 scale

### 2. `group_axis=1`

沿 `K` 轴做 grouped quant：

- 输入激活已经被 pack 成 `[k_group_num * M, k_group_size]`
- kernel 对每个 `K-group` 分别执行 int8 matmul
- vector 阶段对各个 group 的结果累加，得到最终 `[M, N]` 输出

### 3. `group_axis=2`

`expert x K-group` 混合模式：

- `groupList` 记录每个 expert 的 token 数
- `scale` 形状为 `[expert_num * k_group_num, N]`
- 每个 expert 内部独立完成 K-group matmul 与累加

## 实现链路

### AIC 阶段

- `MMCompute()` 从 GM 读取 int8 activation 和 NZ 权重
- 调用 `MatmulImpl` 执行 int8 x int8 -> int32
- 将 partial result 写入 workspace

对于 `group_axis=1/2`，每个 `K-group` 会使用独立的 workspace 槽位，避免
不同 group 的 partial 被覆盖。

### AIV 阶段

- 从 workspace 取出 int32 partial
- `AscendDequant` 乘 `scale[group, n]` 做反量化
- broadcast `perTokenScale`
- 若为 `group_axis=1/2`，先将各个 `K-group` 的反量化结果累加，再输出

### 输出语义

当前主 float16 输出路径不再做 `Gelu`。最终输出是 dequant 和 per-token scale
处理后的 matmul 结果，语义更接近 `torch_npu.npu_quant_matmul`。

## tiling 结构

`quant_group_matmul_custom_tiling.h` 当前除了基础 matmul tiling 外，还包含：

- `groupAxis`
- `kGroupSize`
- `kGroupNum`
- `enableTilePipeline`
- `tileDynamicSchedule`

这些字段用于支持：

- 从 `M` 轴样例平滑扩展到 `K` 轴 grouped matmul
- dynamic quant 与 matmul 的实验性融合
- shared `M` tile / multi-`N` consumer 这类 tile schedule 实验

## dynamic 路径说明

当前目录中存在多条动态量化相关路径：

- `group_dynamic_quant_custom.cpp`
  单独的动态量化 correctness prototype，输入 bf16，输出 int8 和 per-token scale。
- `quant_group_matmul_dynamic_custom`
  单 kernel 风格入口，在 workspace 内部组织 `x_q` / `perTokenScale` / matmul workspace。
- `quant_group_matmul_external_dynamic_custom`
  动态量化结果通过外部张量暴露出来，方便调试。
- `quant_group_matmul_tile_dynamic_custom`
  tile 级动态量化实验入口，目标是减少完整 `x_q` 物化。

这些 dynamic 路径更偏实验和优化验证，稳定基线仍然是显式 `x_int8 + pertoken_scale`
输入的 `quant_group_matmul_custom`。

## host 调用方式

1. CPU 调测
   通过 `ICPU_RUN_KF` 运行 kernel。
2. NPU 调用
   通过 `ACLRT_LAUNCH_KERNEL` 发起 kernel。

`ASCENDC_CPU_DEBUG` 宏用于区分 CPU 调测和 NPU 真机路径。

## 运行样例算子

### 1. 进入目录

```bash
cd custom_op/quant_group_matmul
```

### 2. 配置环境变量

根据当前环境设置 `ASCEND_INSTALL_PATH`，例如：

```bash
export ASCEND_INSTALL_PATH=/usr/local/Ascend/cann-8.5.1
```

### 3. 编译并安装 kernel 库

```bash
bash run.sh -r npu -v Ascend910B3
```

参数说明：

- `RUN_MODE`：`cpu` / `sim` / `npu`
- `SOC_VERSION`：例如 `Ascend910B1`、`Ascend910B2`、`Ascend910B3`、`Ascend910B4`

默认情况下 `run.sh` 只构建并安装 kernel 库。若要连同 sample host bin 一起构建，
可设置 `BUILD_SAMPLE_BIN=1`。

## 更新说明

| 时间 | 更新事项 |
| --- | --- |
| 2026/07/01 | README 更新为当前 kernel 实际实现，补充 group_axis、dynamic 与 tile 实验路径说明 |

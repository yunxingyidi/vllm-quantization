#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnn_gmm1_vecin_fullk.h"

namespace {
constexpr int64_t K = 2048;

template <typename T>
aclTensor *MakeTensor(const std::vector<T> &host, const std::vector<int64_t> &shape,
                      aclDataType dtype, aclFormat format, void **device,
                      const std::vector<int64_t> &storageShape = {})
{
    const size_t bytes = host.size() * sizeof(T);
    if (aclrtMalloc(device, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return nullptr;
    if (aclrtMemcpy(*device, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) return nullptr;
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    const auto &physical = storageShape.empty() ? shape : storageShape;
    return aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0,
                           format, physical.data(), physical.size(), *device);
}

struct TestCase {
    std::string name;
    int64_t m;
    int64_t fullN;
    int64_t kGroupNum;
    std::vector<int64_t> groupEnds;
    int8_t xValue;
    std::vector<int8_t> weightValues;
    float xScale;
    std::vector<float> weightScales;
    bool randomized;
    bool expectFp16Overflow;
    int iterations;
};

int8_t RandomInt8(uint32_t seed)
{
    seed ^= seed >> 16;
    seed *= 0x7feb352dU;
    seed ^= seed >> 15;
    seed *= 0x846ca68bU;
    seed ^= seed >> 16;
    return static_cast<int8_t>(static_cast<int32_t>(seed % 5U) - 2);
}

size_t NzIndex(int64_t k, int64_t n)
{
    return static_cast<size_t>((((n / 32) * (K / 16) + k / 16) * 16 + k % 16) * 32 + n % 32);
}

uint64_t EncodeFixpipeScale(float scale)
{
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(scale));
    std::memcpy(&bits, &scale, sizeof(bits));
    // Same VDEQF16 encoding produced by torch_npu.npu_trans_quant_param.
    return (0x4000ULL << 32) | bits;
}

float FixpipeFp16Scaled(int32_t value, float scale)
{
    constexpr float FP16_MAX = 65504.0F;
    const float scaled = std::max(-FP16_MAX, std::min(FP16_MAX,
        static_cast<float>(value) * scale));
    return aclFloat16ToFloat(aclFloatToFloat16(scaled));
}

int ExpertForRow(const TestCase &test, int64_t row)
{
    for (size_t expert = 0; expert < test.groupEnds.size(); ++expert) {
        if (row < test.groupEnds[expert]) return static_cast<int>(expert);
    }
    return -1;
}

bool RunCase(const TestCase &test, aclrtStream stream)
{
    const int64_t outN = test.fullN / 2;
    const int64_t kGroupSize = K / test.kGroupNum;
    bool passed = false;
    void *xDev = nullptr, *xsDev = nullptr, *gDev = nullptr;
    void *yDev = nullptr, *ysDev = nullptr, *workspace = nullptr;
    aclTensor *x = nullptr, *weightScale = nullptr, *xScale = nullptr, *groupList = nullptr;
    aclTensor *y = nullptr, *yScale = nullptr;
    aclTensorList *weights = nullptr;
    std::vector<void *> weightDev(test.groupEnds.size(), nullptr);
    void *weightScaleDev = nullptr;
    std::vector<aclTensor *> weightTensors(test.groupEnds.size(), nullptr);
    std::vector<int8_t> xHost(test.m * K, test.xValue);
    std::vector<float> xScaleHost(test.m, test.xScale);
    std::vector<std::vector<int8_t>> weightHosts(test.groupEnds.size());
    std::vector<std::vector<float>> weightScaleHosts(test.groupEnds.size());
    std::vector<int64_t> packedWeightScaleHost(
        test.groupEnds.size() * test.kGroupNum * test.fullN);

    if (test.randomized) {
        for (int64_t row = 0; row < test.m; ++row) {
            xScaleHost[row] = 0.004F + static_cast<float>(row % 5) * 0.001F;
            for (int64_t k = 0; k < K; ++k) {
                xHost[row * K + k] = RandomInt8(static_cast<uint32_t>(row * K + k + 17));
            }
        }
    }
    for (size_t expert = 0; expert < test.groupEnds.size(); ++expert) {
        weightHosts[expert].assign(test.fullN * K, test.weightValues[expert]);
        weightScaleHosts[expert].resize(test.kGroupNum * test.fullN);
        for (int64_t group = 0; group < test.kGroupNum; ++group) {
            for (int64_t n = 0; n < test.fullN; ++n) {
                weightScaleHosts[expert][group * test.fullN + n] = test.randomized
                    ? 0.0004F + static_cast<float>((n + group * 3 + expert * 5) % 7) * 0.0001F
                    : test.weightScales[expert] *
                          (1.0F + static_cast<float>(group % 5) * 0.125F);
            }
        }
        if (!test.randomized) continue;
        for (int64_t n = 0; n < test.fullN; ++n) {
            for (int64_t k = 0; k < K; ++k) {
                weightHosts[expert][NzIndex(k, n)] = RandomInt8(
                    static_cast<uint32_t>(expert * test.fullN * K + n * K + k + 101));
            }
        }
    }
    for (size_t expert = 0; expert < test.groupEnds.size(); ++expert) {
        const size_t expertOffset = expert * test.kGroupNum * test.fullN;
        for (size_t index = 0; index < weightScaleHosts[expert].size(); ++index) {
            packedWeightScaleHost[expertOffset + index] = static_cast<int64_t>(
                EncodeFixpipeScale(weightScaleHosts[expert][index]));
        }
    }

    do {
        x = MakeTensor(xHost, {test.m, K}, ACL_INT8, ACL_FORMAT_ND, &xDev);
        xScale = MakeTensor(xScaleHost, {test.m},
                            ACL_FLOAT, ACL_FORMAT_ND, &xsDev);
        groupList = MakeTensor(test.groupEnds, {static_cast<int64_t>(test.groupEnds.size())},
                               ACL_INT64, ACL_FORMAT_ND, &gDev);
        y = MakeTensor(std::vector<int8_t>(test.m * outN), {test.m, outN},
                       ACL_INT8, ACL_FORMAT_ND, &yDev);
        yScale = MakeTensor(std::vector<float>(test.m), {test.m},
                            ACL_FLOAT, ACL_FORMAT_ND, &ysDev);
        if (!x || !xScale || !groupList || !y || !yScale) break;

        weightScale = MakeTensor(
            packedWeightScaleHost,
            {static_cast<int64_t>(test.groupEnds.size()), test.kGroupNum, test.fullN},
            ACL_INT64, ACL_FORMAT_ND, &weightScaleDev);
        if (!weightScale) break;

        for (size_t expert = 0; expert < test.groupEnds.size(); ++expert) {
            weightTensors[expert] = MakeTensor(
                weightHosts[expert], {K, test.fullN},
                ACL_INT8, ACL_FORMAT_FRACTAL_NZ, &weightDev[expert],
                {test.fullN / 32, 128, 16, 32});
            if (!weightTensors[expert]) break;
        }
        bool tensorsReady = true;
        for (size_t expert = 0; expert < test.groupEnds.size(); ++expert) {
            tensorsReady = tensorsReady && weightTensors[expert];
        }
        if (!tensorsReady) break;
        std::vector<const aclTensor *> weightItems(weightTensors.begin(), weightTensors.end());
        weights = aclCreateTensorList(weightItems.data(), weightItems.size());
        if (!weights) break;

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        auto ret = aclnnGmm1VecinFullkGetWorkspaceSize(
            x, weights, weightScale, xScale, groupList, y, yScale,
            &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
            std::cerr << test.name << ": GetWorkspaceSize failed: " << ret << '\n';
            break;
        }
        if (workspaceSize &&
            aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) break;
        double elapsed = 0.0;
        if (test.iterations > 1) {
            ret = aclnnGmm1VecinFullk(workspace, workspaceSize, executor, stream);
            if (ret == ACL_SUCCESS) ret = aclrtSynchronizeStream(stream);
        }
        const auto start = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < test.iterations && ret == ACL_SUCCESS; ++iteration) {
            if (test.iterations > 1) {
                uint64_t nextWorkspaceSize = 0;
                executor = nullptr;
                ret = aclnnGmm1VecinFullkGetWorkspaceSize(
                    x, weights, weightScale, xScale, groupList, y, yScale,
                    &nextWorkspaceSize, &executor);
                if (ret != ACL_SUCCESS || nextWorkspaceSize > workspaceSize) break;
            }
            ret = aclnnGmm1VecinFullk(workspace, workspaceSize, executor, stream);
        }
        const auto syncRet = ret == ACL_SUCCESS ? aclrtSynchronizeStream(stream) : ACL_SUCCESS;
        elapsed = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count();
        if (ret != ACL_SUCCESS || syncRet != ACL_SUCCESS) {
            std::cerr << test.name << ": kernel execution failed: launch=" << ret
                      << ", sync=" << syncRet;
            const char *recent = aclGetRecentErrMsg();
            if (recent != nullptr) std::cerr << ", detail=" << recent;
            std::cerr << '\n';
            break;
        }

        std::vector<int8_t> actualY(test.m * outN);
        std::vector<float> actualScale(test.m);
        if (aclrtMemcpy(actualY.data(), actualY.size(), yDev, actualY.size(),
                        ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
            aclrtMemcpy(actualScale.data(), actualScale.size() * sizeof(float), ysDev,
                        actualScale.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) break;

        if (test.expectFp16Overflow) {
            const int64_t dot = K * static_cast<int64_t>(test.xValue) * test.weightValues[0];
            const float z = static_cast<float>(dot) * test.xScale * test.weightScales[0];
            const float post = (z / (1.0F + std::exp(-z))) * z;
            const float int32GoldenScale = post / 127.0F;
            const float fixpipeZ = FixpipeFp16Scaled(
                static_cast<int32_t>(dot), test.weightScales[0]) * test.xScale;
            const float fixpipePost =
                (fixpipeZ / (1.0F + std::exp(-fixpipeZ))) * fixpipeZ;
            const float fixpipeGoldenScale = fixpipePost / 127.0F;
            const float observed = actualScale[0];
            bool prescaleOk = std::isfinite(observed) &&
                std::abs(observed - fixpipeGoldenScale) <= fixpipeGoldenScale * 0.02F + 1.0e-8F;
            for (auto value : actualY) prescaleOk = prescaleOk && value == 127;
            if (!prescaleOk) {
                std::cerr << test.name << ": unexpected pre-scaled FP16 result; observed scale="
                          << observed << ", fixpipe-golden=" << fixpipeGoldenScale
                          << ", y[0]=" << static_cast<int>(actualY[0]) << '\n';
                break;
            }
            std::cout << "PASS " << test.name << "; pre-scaled FP16 scale=" << observed
                      << ", INT32-golden scale=" << int32GoldenScale << '\n';
            passed = true;
            break;
        }

        if (test.randomized) {
            size_t withinOne = 0;
            int maxQuantError = 0;
            double mere = 0.0;
            double mare = 0.0;
            for (int64_t row = 0; row < test.m; ++row) {
                const int expert = ExpertForRow(test, row);
                std::vector<float> golden(outN);
                float absMax = 0.0F;
                for (int64_t n = 0; n < outN; ++n) {
                    float up = 0.0F;
                    float gate = 0.0F;
                    for (int64_t group = 0; group < test.kGroupNum; ++group) {
                        int32_t upDot = 0;
                        int32_t gateDot = 0;
                        for (int64_t k = group * kGroupSize;
                             k < (group + 1) * kGroupSize; ++k) {
                            const int32_t xv = xHost[row * K + k];
                            upDot += xv * weightHosts[expert][NzIndex(k, n)];
                            gateDot += xv * weightHosts[expert][NzIndex(k, outN + n)];
                        }
                        up += FixpipeFp16Scaled(upDot,
                            weightScaleHosts[expert][group * test.fullN + n]);
                        gate += FixpipeFp16Scaled(gateDot,
                            weightScaleHosts[expert][group * test.fullN + outN + n]);
                    }
                    up *= xScaleHost[row];
                    gate *= xScaleHost[row];
                    golden[n] = (gate / (1.0F + std::exp(-gate))) * up;
                    absMax = std::max(absMax, std::abs(golden[n]));
                }
                const float goldenScale = std::max(absMax, 1.0e-6F) / 127.0F;
                const double relative = std::abs(actualScale[row] - goldenScale) /
                    (std::abs(goldenScale) + 1.0e-7F);
                mere += relative;
                mare = std::max(mare, relative);
                for (int64_t n = 0; n < outN; ++n) {
                    const int expected = static_cast<int>(std::nearbyint(
                        std::max(-127.0F, std::min(127.0F, golden[n] / goldenScale))));
                    const int error = std::abs(static_cast<int>(actualY[row * outN + n]) - expected);
                    maxQuantError = std::max(maxQuantError, error);
                    if (error <= 1) ++withinOne;
                }
            }
            mere /= test.m;
            const double withinOneRatio = static_cast<double>(withinOne) / actualY.size();
            constexpr double FP32_THRESHOLD = 1.0 / 8192.0;
            if (mere >= FP32_THRESHOLD || mare >= 10.0 * FP32_THRESHOLD ||
                withinOneRatio < 0.99 || maxQuantError > 2) {
                std::cerr << test.name << ": precision failed; MERE=" << mere
                          << ", MARE=" << mare << ", y<=1 ratio=" << withinOneRatio
                          << ", max y error=" << maxQuantError << '\n';
                break;
            }
            std::cout << "PASS " << test.name << "; MERE=" << mere
                      << ", MARE=" << mare << ", y<=1 ratio=" << withinOneRatio << '\n';
            passed = true;
            break;
        }

        bool valuesOk = true;
        for (int64_t row = 0; row < test.m && valuesOk; ++row) {
            const int expert = ExpertForRow(test, row);
            float z = 0.0F;
            const int32_t groupDot = static_cast<int32_t>(
                kGroupSize * static_cast<int64_t>(test.xValue) * test.weightValues[expert]);
            for (int64_t group = 0; group < test.kGroupNum; ++group) {
                z += FixpipeFp16Scaled(groupDot,
                    weightScaleHosts[expert][group * test.fullN]);
            }
            z *= test.xScale;
            const float post = (z / (1.0F + std::exp(-z))) * z;
            const float expectedScale = post / 127.0F;
            for (int64_t column = 0; column < outN; ++column) {
                if (actualY[row * outN + column] != 127) {
                    std::cerr << test.name << ": unexpected y at row=" << row
                              << ", column=" << column << ": "
                              << static_cast<int>(actualY[row * outN + column]) << '\n';
                    valuesOk = false;
                    break;
                }
            }
            if (std::abs(actualScale[row] - expectedScale) > expectedScale * 0.02F + 1.0e-8F) {
                std::cerr << test.name << ": unexpected scale at row=" << row
                          << ": " << actualScale[row] << ", expected=" << expectedScale << '\n';
                valuesOk = false;
            }
        }
        if (!valuesOk) break;
        std::cout << "PASS " << test.name << "; M=" << test.m
                  << ", experts=" << test.groupEnds.size();
        if (test.iterations > 1) {
            std::cout << ", ACLNN avg=" << elapsed / test.iterations << " us";
        }
        std::cout << '\n';
        passed = true;
    } while (false);

    if (weights) aclDestroyTensorList(weights);
    for (auto tensor : weightTensors) if (tensor) aclDestroyTensor(tensor);
    for (auto tensor : {x, weightScale, xScale, groupList, y, yScale}) if (tensor) aclDestroyTensor(tensor);
    for (auto ptr : weightDev) if (ptr) aclrtFree(ptr);
    if (weightScaleDev) aclrtFree(weightScaleDev);
    for (auto ptr : {xDev, xsDev, gDev, yDev, ysDev, workspace}) if (ptr) aclrtFree(ptr);
    return passed;
}
}  // namespace

int main(int argc, char **argv)
{
    aclrtStream stream = nullptr;
    if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(0) != ACL_SUCCESS ||
        aclrtCreateStream(&stream) != ACL_SUCCESS) return 1;

    const std::vector<TestCase> tests = {
        {"single-expert-full-tile", 16, 1536, 1, {16}, 1, {1}, 0.01F, {0.001F}, false, false, 1},
        {"single-expert-m-tail", 7, 1536, 1, {7}, 1, {1}, 0.01F, {0.001F}, false, false, 1},
        {"multi-expert-boundary", 19, 1536, 1, {3, 19}, 1, {1, 2}, 0.01F,
         {0.001F, 0.001F}, false, false, 1},
        {"random-nz-precision", 5, 1536, 1, {2, 5}, 0, {0, 0}, 0.0F,
         {0.0F, 0.0F}, true, false, 1},
        {"tp2-n-tail", 7, 768, 1, {7}, 1, {1}, 0.01F, {0.001F}, false, false, 1},
        {"tp4-n-tail", 7, 384, 1, {7}, 1, {1}, 0.01F, {0.001F}, false, false, 1},
        {"tp8-reduce-tail", 7, 192, 1, {7}, 1, {1}, 0.01F, {0.001F}, false, false, 1},
        {"tp1-kgroup32-random", 3, 1536, 64, {3}, 0, {0}, 0.0F, {0.0F}, true, false, 1},
        {"tp2-kgroup32-random", 3, 768, 64, {3}, 0, {0}, 0.0F, {0.0F}, true, false, 1},
        {"tp4-kgroup32-random", 3, 384, 64, {3}, 0, {0}, 0.0F, {0.0F}, true, false, 1},
        {"tp8-kgroup32-random", 3, 192, 64, {3}, 0, {0}, 0.0F, {0.0F}, true, false, 1},
        {"fp16-fixpipe-overflow", 1, 1536, 1, {1}, 127, {127}, 0.001F, {0.001F}, false, true, 1},
        {"benchmark-full-tile", 16, 1536, 1, {16}, 1, {1}, 0.01F, {0.001F}, false, false, 50},
        {"benchmark-tp2-kgroup32", 16, 768, 64, {16}, 1, {1}, 0.01F,
         {0.001F}, false, false, 50},
        {"benchmark-tp8-kgroup32", 16, 192, 64, {16}, 1, {1}, 0.01F,
         {0.001F}, false, false, 50},
    };

    const std::string selected = argc > 1 ? argv[1] : tests.front().name;
    const TestCase *test = nullptr;
    for (const auto &candidate : tests) {
        if (candidate.name == selected) test = &candidate;
    }
    if (test == nullptr) {
        std::cerr << "unknown test case: " << selected << '\n';
        aclrtDestroyStream(stream);
        aclrtResetDevice(0);
        aclFinalize();
        return 2;
    }
    const bool passed = RunCase(*test, stream);

    if (stream) aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return passed ? 0 : 1;
}

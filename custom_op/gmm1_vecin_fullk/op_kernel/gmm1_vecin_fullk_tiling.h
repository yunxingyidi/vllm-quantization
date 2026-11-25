#ifndef GMM1_VECIN_FULLK_TILING_H
#define GMM1_VECIN_FULLK_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

// N is derived from the local TP weight. K may be evaluated as one full-K
// group or as multiple scale groups; A/B retain their original full-K stride.
BEGIN_TILING_DATA_DEF(Gmm1VecInFullKTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalM);
  TILING_DATA_FIELD_DEF(uint32_t, expertNum);
  TILING_DATA_FIELD_DEF(uint32_t, coreNum);
  TILING_DATA_FIELD_DEF(uint32_t, baseM);
  TILING_DATA_FIELD_DEF(uint32_t, baseN);
  TILING_DATA_FIELD_DEF(uint32_t, k);
  TILING_DATA_FIELD_DEF(uint32_t, fullN);
  TILING_DATA_FIELD_DEF(uint32_t, halfN);
  TILING_DATA_FIELD_DEF(uint32_t, kGroupNum);
  TILING_DATA_FIELD_DEF(uint32_t, kGroupSize);
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, gateTiling);
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, upTiling);
  // 0: legacy epilogue, 1: single-group NZ, 2: streamed grouped NZ.
  TILING_DATA_FIELD_DEF(uint32_t, nzEpilogue);
  // Bit 0: legacy NZ accumulation; bit 1: reuse unchanged OrgShape.
  TILING_DATA_FIELD_DEF(uint32_t, reserved);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Gmm1VecinFullk, Gmm1VecInFullKTilingData)

}  // namespace optiling

#endif

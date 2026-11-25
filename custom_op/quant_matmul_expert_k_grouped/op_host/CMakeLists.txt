add_op_to_compiled_list()

if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        quant_matmul_expert_k_grouped_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME QuantMatmulExpertKGrouped
    OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -Werror
)

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE quant_matmul_expert_k_grouped ACLNNTYPE aclnn_exclude)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

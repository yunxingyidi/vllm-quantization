add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE gmm1_vecin_fullk_def.cpp)
endif()

add_ops_compile_options(
    OP_NAME Gmm1VecinFullk
    OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations -Werror
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE gmm1_vecin_fullk ACLNNTYPE aclnn_exclude)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../op_kernel)
endif()

# NNOP canonicalizes the digit in Gmm1VecinFullk as `gmm_1_vecin_fullk`,
# while the kernel build directory follows this project's `gmm1_...` name.
# Ship the canonical alias so ACLNN can locate the generated binary config.
install(CODE [[
    set(_config_dir "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/config/ascend910b")
    file(MAKE_DIRECTORY "${_config_dir}")
    execute_process(COMMAND "${CMAKE_COMMAND}" -E create_symlink
        gmm1_vecin_fullk.json "${_config_dir}/gmm_1_vecin_fullk.json")
]])

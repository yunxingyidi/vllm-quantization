cmake_minimum_required(VERSION 3.16)
project(vllm_smooth_custom_ops LANGUAGES NONE)

set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "CANN installation")

# The ACLNN registry operators share one vendor package/libcust_opapi.so.
add_custom_target(registry_ops
    COMMAND ${CMAKE_COMMAND} -E env
        ASCEND_HOME_PATH=${ASCEND_CANN_PACKAGE_PATH}
        bash "${CMAKE_CURRENT_SOURCE_DIR}/scripts/build_groupquant_moe_fused.sh"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Building the combined GroupQuant ACLNN registry package"
    USES_TERMINAL
)

add_custom_target(all_ops DEPENDS registry_ops)

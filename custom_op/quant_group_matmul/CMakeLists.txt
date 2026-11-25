cmake_minimum_required(VERSION 3.16)
project(Ascend_c)

set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
set(SOC_VERSION "Ascend310P3" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE STRING "ASCEND CANN package installation directory"
)

# Some CANN layouts place headers under:
#   <ASCEND_CANN_PACKAGE_PATH>/aarch64-linux/include
# while ascendc internally includes them as "register/xxx.h".
# Add this include root explicitly to avoid missing-header failures.
if(EXISTS "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/include")
    include_directories("${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/include")
endif()
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "path for install()" FORCE)
endif()
option(BUILD_ASCENDC_SAMPLE_BIN "Build sample host executable ascendc_kernels_bbit" OFF)

# ${KERNEL_FILES} are used to compile ascendc kernel library.
# Keep tiling cpp out of this list; it is host-side and linked separately.
set(KERNEL_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/quant_group_matmul_custom.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/quant_group_matmul_tile_dynamic_custom.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/quant_group_matmul_custom_int8.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/group_dynamic_quant_custom.cpp
)

if("${RUN_MODE}" STREQUAL "cpu")
    include(cmake/cpu_lib.cmake)
elseif("${RUN_MODE}" STREQUAL "sim" OR "${RUN_MODE}" STREQUAL "npu")
    include(cmake/npu_lib.cmake)
else()
    message("invalid RUN_MODE: ${RUN_MODE}")
endif()

if(BUILD_ASCENDC_SAMPLE_BIN)
    add_executable(ascendc_kernels_bbit
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/quant_group_matmul_custom_tiling.cpp
    )

    target_compile_options(ascendc_kernels_bbit PRIVATE
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:-g>>
        -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Werror
    )

    target_link_libraries(ascendc_kernels_bbit PRIVATE
        $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:host_intf_pub>>
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:ascendcl>>
        ascendc_kernels_${RUN_MODE}
        dl
    )

    install(TARGETS ascendc_kernels_bbit
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

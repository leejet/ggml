if (NOT EXISTS $ENV{ROCM_PATH})
    if (NOT EXISTS /opt/rocm)
        set(ROCM_PATH /usr)
    else()
        set(ROCM_PATH /opt/rocm)
    endif()
else()
    set(ROCM_PATH $ENV{ROCM_PATH})
endif()

list(APPEND CMAKE_PREFIX_PATH  ${ROCM_PATH})
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}/lib64/cmake")

if (NOT DEFINED CMAKE_HIP_FLAGS_DEBUG)
    set(CMAKE_HIP_FLAGS_DEBUG "-g -O2")
endif()

# CMake on Windows doesn't support the HIP language yet
if (WIN32)
    set(CXX_IS_HIPCC TRUE)
else()
    string(REGEX MATCH "hipcc(\.bat)?$" CXX_IS_HIPCC "${CMAKE_CXX_COMPILER}")
endif()

if (CXX_IS_HIPCC)
    if (LINUX)
        if (NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
            message(WARNING "Only LLVM is supported for HIP, hint: CXX=/opt/rocm/llvm/bin/clang++")
        endif()

        message(WARNING "Setting hipcc as the C++ compiler is legacy behavior."
                " Prefer setting the HIP compiler directly. See README for details.")
    endif()
else()
    # Forward (AMD)GPU_TARGETS to CMAKE_HIP_ARCHITECTURES.
    if(AMDGPU_TARGETS AND NOT GPU_TARGETS)
        set(GPU_TARGETS ${AMDGPU_TARGETS})
    endif()
    if(GPU_TARGETS AND NOT CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES ${GPU_TARGETS})
    endif()
    cmake_minimum_required(VERSION 3.21)
    enable_language(HIP)
endif()

find_package(hip     REQUIRED)
find_package(hipblas REQUIRED)
find_package(rocblas REQUIRED)

if (GGML_HIP_RCCL)
    find_package(rccl REQUIRED)
endif()

if (${hip_VERSION} VERSION_LESS 6.1)
    message(FATAL_ERROR "At least ROCM/HIP V6.1 is required")
endif()

message(STATUS "HIP and hipBLAS found")

file(GLOB   GGML_HEADERS_ROCM "../ggml-cuda/*.cuh")
list(APPEND GGML_HEADERS_ROCM "../../include/ggml-cuda.h")

file(GLOB   GGML_SOURCES_ROCM "../ggml-cuda/*.cu")
file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-tile*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})
file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-mma*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})
file(GLOB   SRCS "../ggml-cuda/template-instances/mmq*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})
file(GLOB   SRCS "../ggml-cuda/template-instances/mmf*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})

if (GGML_CUDA_FA_ALL_QUANTS)
    file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-vec*.cu")
    list(APPEND GGML_SOURCES_ROCM ${SRCS})
    add_compile_definitions(GGML_CUDA_FA_ALL_QUANTS)
else()
    list(APPEND GGML_SOURCES_ROCM
        ../ggml-cuda/template-instances/fattn-vec-instance-f16-f16.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-q4_0-q4_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-q8_0-q8_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-bf16-bf16.cu)
endif()

# -mcumode: scalar-unit mode splits the scalar pipeline per wave; benefits
# LDS-heavy MMQ kernels (shared LDS/SALU pressure in wave32). Applied to the
# w4 convrot kernel file only - a global flag would change every kernel.
# -mwavefrontsize64: clang's gfx1100 default is wave32 (verified via
# clang-offload-bundler + llvm-readobj); wave64 halves the waves per block
# (4 vs 8), doubling per-wave ILP scope. Bench A/B against the wave32 build.
# GFX10 GUARD (2026-09-18): ROCm 10.0 clang 23 segfaults in LLVM
# BranchRelaxation (SIInstrInfo::getInstSizeInBytes) compiling this
# kernel's wave64 codegen for gfx1030 - at any opt level, with or without
# -mcumode; wave32 compiles clean (verified on the failing WSL build).
# Tuning was measured on gfx1100 only; gfx10 targets keep the compiler
# default (wave32) until the compiler bug is fixed and the win is
# re-benched on RDNA2.
if (NOT CMAKE_HIP_ARCHITECTURES MATCHES "gfx10")
    set_source_files_properties(../ggml-cuda/mmq-w4-convrot.cu PROPERTIES
        COMPILE_OPTIONS "-mcumode;-mwavefrontsize64")
endif()

# w4a8 WMMA (gfx11): benched 2.08x SLOWER than dp4a on 7900 XTX (5.69
# vs 2.74 s/it) - RDNA3's wmma iu8 shares one matrix pipe per WGP and
# cannot co-execute with VALU, so a block's 4 waves serialize on it. Keep
# disabled for gfx11. The dispatch skeleton in mmq-w4-convrot.cu
# (GGML_W4A8_WMMA_GFX11 define + runtime cc guard) stays for the gfx12
# port below, where the matrix pipe is 4x wider and VALU co-executes.
# set_property(SOURCE ../ggml-cuda/mmq-w4-convrot.cu APPEND PROPERTY
#     COMPILE_OPTIONS "-DGGML_W4A8_WMMA_GFX11=1")

# w4a8 WMMA (gfx12/RDNA4): port of the gfx11 kernel to the gfx12 wave64
# intrinsic - each lane supplies one 4-byte A/B slice (no replicated rows),
# so decode work per wave drops 4x vs gfx11. Runs under this file's
# -mwavefrontsize64 (unlike the w32 gfx12 intrinsic, which needs
# wavefrontsize32 and would fight the wave64 tuning above). Compile-
# verified on gfx1201 only: no RDNA4 hardware was available for the fixture
# gate or a bench. Auto-selected when the build carries a gfx12 target
# (runtime cc guard keeps other GPUs on dp4a); the one-time "w4a8 convrot
# kernel:" debug log names the selected path.
if (CMAKE_HIP_ARCHITECTURES MATCHES "gfx12")
    set_property(SOURCE ../ggml-cuda/mmq-w4-convrot.cu APPEND PROPERTY
        COMPILE_OPTIONS "-DGGML_W4A8_WMMA_GFX12=1")
endif()

ggml_add_backend_library(ggml-hip
                         ${GGML_HEADERS_ROCM}
                         ${GGML_SOURCES_ROCM}
                        )

# TODO: do not use CUDA definitions for HIP
if (NOT GGML_BACKEND_DL)
    target_compile_definitions(ggml PUBLIC GGML_USE_CUDA)
endif()

add_compile_definitions(GGML_USE_HIP)

if (GGML_CUDA_FORCE_MMQ)
    add_compile_definitions(GGML_CUDA_FORCE_MMQ)
endif()

if (GGML_CUDA_FORCE_CUBLAS)
    add_compile_definitions(GGML_CUDA_FORCE_CUBLAS)
endif()

if (GGML_CUDA_NO_PEER_COPY)
    add_compile_definitions(GGML_CUDA_NO_PEER_COPY)
endif()

if (GGML_HIP_GRAPHS)
    add_compile_definitions(GGML_HIP_GRAPHS)
endif()

if (GGML_HIP_NO_VMM)
    add_compile_definitions(GGML_HIP_NO_VMM)
endif()

if (NOT GGML_HIP_MMQ_MFMA)
    add_compile_definitions(GGML_HIP_NO_MMQ_MFMA)
endif()

if (GGML_HIP_RCCL)
    add_compile_definitions(GGML_USE_NCCL) # RCCL has the same interface as NCCL.
endif()

if (GGML_HIP_EXPORT_METRICS)
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Rpass-analysis=kernel-resource-usage --save-temps")
endif()

# Fast math for HIP, like CUDA's -use_fast_math. Not -ffast-math: that implies -ffinite-math-only, which breaks ggml's INFINITY masking and produces NaNs.
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -funsafe-math-optimizations")

if (NOT GGML_CUDA_FA)
    add_compile_definitions(GGML_CUDA_NO_FA)
endif()

if (CXX_IS_HIPCC)
    set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES LANGUAGE CXX)
    if (WIN32 AND CMAKE_BUILD_TYPE STREQUAL "Debug")
        # CMake on Windows doesn't support the HIP language yet.
        # Therefore we workaround debug build's failure on HIP backend this way.
        set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES COMPILE_FLAGS "-O2 -g")
    endif()
    target_link_libraries(ggml-hip PRIVATE hip::device)
else()
    set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES LANGUAGE HIP)
endif()

if (GGML_STATIC)
    message(FATAL_ERROR "Static linking not supported for HIP/ROCm")
endif()

if (GGML_HIP_RCCL)
    target_link_libraries(ggml-hip PRIVATE ggml-base roc::rccl)
endif()

target_link_libraries(ggml-hip PRIVATE ggml-base hip::host roc::rocblas roc::hipblas)

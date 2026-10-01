# engines/qwen35moe - the Qwen3.5/3.6 MoE (qwen35moe) engine: its own CMake project, so the upstream engine's build
# (the repository root's CMakeLists.txt) stays as it is.  setup_qwen36.py builds it with Ninja:
#   cmake -G Ninja -S engines/qwen35moe -B build-qwen35moe -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)   # the GPUs in this PC; setup passes the exact list
endif()
project(strata_qwen35moe VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

add_compile_definitions(SQ_VERSION="${PROJECT_VERSION}")
if(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CXX>:/O2>
                      $<$<COMPILE_LANGUAGE:CXX>:/EHsc> $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus> $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>)
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
  # the expert kernels per instruction set (cpu_moe.cpp picks one at run time; everything else stays x86-64)
  set(AVX512_FLAGS /arch:AVX512)
  set(AVX2_FLAGS /arch:AVX2)
else()
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-O3>)
  set(AVX512_FLAGS -mavx512f -mavx512bw -mavx512vl -mavx512vnni -mavx512dq -mfma -mf16c)
  set(AVX2_FLAGS -mavx2 -mfma -mf16c)
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -lineinfo --expt-relaxed-constexpr -Xcudafe --diag_suppress=177")
if(MSVC)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/utf-8 -Xcompiler=/wd4819 -Xcompiler=/Zc:preprocessor")
endif()

set_source_files_properties(src/cpu_avx512.cpp PROPERTIES COMPILE_OPTIONS "${AVX512_FLAGS}")
set_source_files_properties(src/cpu_avx2.cpp PROPERTIES COMPILE_OPTIONS "${AVX2_FLAGS}")

add_library(sq_cpu STATIC src/gguf.cpp src/cpu_moe.cpp src/cpu_avx512.cpp src/cpu_avx2.cpp)
target_include_directories(sq_cpu PUBLIC src)
target_link_libraries(sq_cpu PUBLIC CUDA::cudart_static Threads::Threads)

add_library(sq_core STATIC
  src/model.cpp
  src/engine.cpp
  src/sampler.cpp
  src/kernels.cu
  src/prefill.cu
)
target_include_directories(sq_core PUBLIC src)
target_link_libraries(sq_core PUBLIC sq_cpu CUDA::cudart_static CUDA::cublas)
set_target_properties(sq_core PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

add_executable(strata-qwen35moe src/main.cpp)
target_link_libraries(strata-qwen35moe PRIVATE sq_core)

add_executable(test_cpu_kernels src/test_cpu_kernels.cpp)
target_link_libraries(test_cpu_kernels PRIVATE sq_cpu)

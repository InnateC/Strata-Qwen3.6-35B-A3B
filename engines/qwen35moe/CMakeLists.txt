cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)   # RTX 50xx (Blackwell consumer)
endif()
project(strataq LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

if(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CXX>:/O2>
                      $<$<COMPILE_LANGUAGE:CXX>:/EHsc> $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus> $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>)
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
  set(CPU_SIMD_FLAGS /arch:AVX512)
else()
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-O3>)
  set(CPU_SIMD_FLAGS -mavx512f -mavx512bw -mavx512vl -mavx512vnni -mavx512dq -mfma -mf16c)
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -lineinfo --expt-relaxed-constexpr -Xcudafe --diag_suppress=177")
if(MSVC)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/utf-8 -Xcompiler=/wd4819 -Xcompiler=/Zc:preprocessor")
endif()

add_library(strataq_cpu STATIC src/gguf.cpp src/cpu_moe.cpp)
target_include_directories(strataq_cpu PUBLIC src)
target_link_libraries(strataq_cpu PUBLIC CUDA::cudart_static)

add_library(strataq_core STATIC
  src/model.cpp
  src/engine.cpp
  src/sampler.cpp
  src/kernels.cu
  src/prefill.cu
)
set_source_files_properties(src/cpu_moe.cpp PROPERTIES COMPILE_OPTIONS "${CPU_SIMD_FLAGS}")
target_include_directories(strataq_core PUBLIC src)
target_link_libraries(strataq_core PUBLIC strataq_cpu CUDA::cudart_static CUDA::cublas)
set_target_properties(strataq_core PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

add_executable(strataq src/main.cpp)
target_link_libraries(strataq PRIVATE strataq_core)

add_executable(test_cpu_kernels src/test_cpu_kernels.cpp)
set_source_files_properties(src/test_cpu_kernels.cpp PROPERTIES COMPILE_OPTIONS "${CPU_SIMD_FLAGS}")
target_link_libraries(test_cpu_kernels PRIVATE strataq_cpu)

cmake_minimum_required(VERSION 3.22)
project(qwen3_rai_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT WIN32)
  message(FATAL_ERROR "Qwen3 corelib tests currently require Windows")
endif()

find_path(RYZENAI_CORELIB_INCLUDE_DIR NAMES ryzenai/corelib.h REQUIRED)
find_path(BOOST_INCLUDE_DIR NAMES boost/program_options.hpp
    HINTS "$ENV{CONDA_PREFIX}/Library/include"
          "$ENV{USERPROFILE}/anaconda3/Library/include" REQUIRED)
find_path(XRT_INCLUDE_DIR NAMES xrt/xrt_bo.h
    HINTS "$ENV{XRT_INCLUDE_DIR}"
          "C:/dev/XRT/src/runtime_src/core/include" REQUIRED)
set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")

# The fake corelib and the GGUF fixture builder are phi4_rai's; the engine
# under test is Qwen3's, built from the same sources flm_rai compiles.
add_executable(test_qwen3_rai
    test_qwen3_rai.cpp
    "${CMAKE_CURRENT_LIST_DIR}/../phi4_rai/fake_corelib.cpp"
    "${FLM_SOURCE_DIR}/common/rai/corelib_api.cpp"
    "${FLM_SOURCE_DIR}/common/rai/corelib_runtime.cpp"
    "${FLM_SOURCE_DIR}/common/rai/gguf_file.cpp"
    "${FLM_SOURCE_DIR}/common/rai/host_ops.cpp"
    "${FLM_SOURCE_DIR}/common/rai/weight_cache.cpp"
    "${FLM_SOURCE_DIR}/common/models/qwen3/rai/aie_next/qwen3_rai_config.cpp"
    "${FLM_SOURCE_DIR}/common/models/qwen3/rai/aie_next/qwen3_rai_gguf.cpp"
    "${FLM_SOURCE_DIR}/common/models/qwen3/rai/aie_next/qwen3_rai_shape_plan.cpp"
    "${FLM_SOURCE_DIR}/common/models/qwen3/rai/aie_next/qwen3_rai.cpp")
target_include_directories(test_qwen3_rai PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}"
    "${FLM_SOURCE_DIR}/include"
    "${RYZENAI_CORELIB_INCLUDE_DIR}"
    "${BOOST_INCLUDE_DIR}"
    "${XRT_INCLUDE_DIR}")
target_compile_definitions(test_qwen3_rai PRIVATE
    RYZENAI_CORELIB_STATIC=1 USEAVX2=1 DISABLE_ABI_CHECK=1
    _ENABLE_EXTENDED_ALIGNED_STORAGE WIN32_LEAN_AND_MEAN NOMINMAX)
target_compile_options(test_qwen3_rai PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/wd4005 /wd4244>)
target_link_directories(test_qwen3_rai PRIVATE "${XRT_INCLUDE_DIR}/../lib")
target_link_libraries(test_qwen3_rai PRIVATE xrt_coreutil)

include(CTest)
add_test(NAME test_qwen3_rai COMMAND test_qwen3_rai)

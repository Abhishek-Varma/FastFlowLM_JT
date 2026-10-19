# d3 contract tests: no NPU, no XRT, so this builds and runs on Linux CI as well as Windows.
#   cmake -S src/test/d3_contract -B build-d3c && cmake --build build-d3c && ctest --test-dir build-d3c
# -DFLM_D3_TEST_TOKENIZER=ON also links tokenizers-cpp (needs cargo) so a local run with the model package
# checks token ids and the answer codes:  test_d3_contract <fixtures> <D3-Lite-NPU2 dir>
cmake_minimum_required(VERSION 3.22)
project(d3_contract_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(FLM_D3_TEST_TOKENIZER "Check token ids with tokenizers-cpp" OFF)

set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")

add_executable(test_d3_contract
    test_d3_contract.cpp
    "${FLM_SOURCE_DIR}/common/decision/py_json.cpp"
    "${FLM_SOURCE_DIR}/common/decision/d3_contract.cpp")
target_include_directories(test_d3_contract PRIVATE "${FLM_SOURCE_DIR}/include")

if(FLM_D3_TEST_TOKENIZER)
    add_subdirectory("${FLM_SOURCE_DIR}/../third_party/tokenizers-cpp" tokenizers EXCLUDE_FROM_ALL)
    target_link_libraries(test_d3_contract PRIVATE tokenizers_cpp)
    target_include_directories(test_d3_contract PRIVATE "${FLM_SOURCE_DIR}/../third_party/tokenizers-cpp/include")
    target_compile_definitions(test_d3_contract PRIVATE FLM_D3_TEST_TOKENIZER)
endif()

enable_testing()
add_test(NAME test_d3_contract
         COMMAND test_d3_contract "${CMAKE_CURRENT_LIST_DIR}/fixtures")

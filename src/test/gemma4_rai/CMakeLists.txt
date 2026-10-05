cmake_minimum_required(VERSION 3.20)
project(gemma4_rai_tests LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()

# FLM_SOURCE_DIR is passed in by the configure command, e.g.:
#   cmake -B build-g4 -S src/test/gemma4_rai -DFLM_SOURCE_DIR=<repo>/src
if(NOT DEFINED FLM_SOURCE_DIR)
    set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")
endif()

include_directories("${FLM_SOURCE_DIR}/include" "${FLM_SOURCE_DIR}/test/phi4_rai")

# Every Gemma 4 test target is declared here up front, each guarded on the
# existence of its own .cpp, so later tasks add only their source file and
# never edit this file -- two agents touching this CMakeLists at once is
# exactly the collision a previous attempt on this project produced.
#
# Order and provenance (see docs/superpowers/plans/2026-09-22-gemma4-rai.md,
# Stage A and Stage C):
#   test_gemma4_host      Task A2 (this task)
#   test_gemma4_frontend  Task A3
#   test_gemma4_config    Task C1
#   test_gemma4_derivations, test_gemma4_gguf  Tasks C2-C5
#   test_gemma4_shape_plan  Task C6
#   test_gemma4_engine    Tasks C7-C9
#   test_gemma4_backend   Task C10

# --- test_gemma4_host (Task A2) -----------------------------------------
# The host layer holds NO corelib types -- that is the whole point of it,
# and it is what lets this target be built and run with no NPU, no XRT, no
# boost and no RYZENAI_CORELIB_INCLUDE_DIR. It links nothing but its own
# .cpp. If this target ever starts needing the corelib header or an XRT/
# boost include path, the host layer has been contaminated: fix the layer,
# not this file.
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/test_gemma4_host.cpp")
    add_executable(test_gemma4_host
        test_gemma4_host.cpp
        "${FLM_SOURCE_DIR}/common/models/gemma4/rai/aie_next/gemma4_rai_host.cpp")
    add_test(NAME test_gemma4_host COMMAND test_gemma4_host)
endif()

# --- test_gemma4_frontend (Task A3) --------------------------------------
# Exercises Gemma4e::engine_config() through the real AutoModel/Gemma4e
# machinery, with a stub causal_lm engine standing in for the rai engine
# library this build does not link -- the same pattern test_phi4_frontend
# uses in src/test/phi4_rai/. It needs XRT and boost headers (xrt/detail/
# any.h pulls in boost/any.hpp) but not the corelib header itself, since the
# frontend never includes rai/corelib_api.hpp directly.
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/test_gemma4_frontend.cpp")
    # Not REQUIRED: a configure run that only wants test_gemma4_host (or any
    # other dependency-free target) must not be blocked by this target's
    # probes. Absence is reported below -- loudly, and as a ctest FAILURE --
    # rather than failing the whole configure.
    #
    # THE FOURTH HINT IS FOR A GIT WORKTREE, and its absence made this the
    # last live instance in this file of the bug the shape-plan and engine
    # blocks were already hardened against. The third hint resolves to
    # <workspace>/xrt_package only when the checkout sits directly in the
    # workspace; a worktree under <repo>/.worktrees/<name> is two levels
    # deeper. Without it this target was silently dropped on a FIRST
    # configure in a worktree -- ctest then reported "100% tests passed ...
    # out of 7" without ever naming it -- and reappeared on a SECOND
    # configure, because the engine block below does have the deeper hint and
    # writes XRT_INCLUDE_DIR into the CMake cache, which this block then
    # picks up on the next pass. Every test count taken from this directory
    # was therefore a function of configure history. Measured both ways
    # before this line was added: 7 targets / 99 PASS on a first configure,
    # 8 / 101 on a second.
    find_path(XRT_INCLUDE_DIR NAMES xrt/xrt_bo.h
        HINTS "$ENV{XRT_INCLUDE_DIR}"
              "${CMAKE_CURRENT_LIST_DIR}/../../../../xrt_package/xrt/include"
              "${CMAKE_CURRENT_LIST_DIR}/../../../../../../xrt_package/xrt/include"
              "C:/dev/XRT/src/runtime_src/core/include")
    find_path(BOOST_INCLUDE_DIR NAMES boost/any.hpp
        HINTS "$ENV{CONDA_PREFIX}/Library/include"
              "$ENV{USERPROFILE}/anaconda3/Library/include")
    if(XRT_INCLUDE_DIR AND BOOST_INCLUDE_DIR)
        add_executable(test_gemma4_frontend
            test_gemma4_frontend.cpp
            "${FLM_SOURCE_DIR}/common/AutoModel/automodel.cpp"
            "${FLM_SOURCE_DIR}/common/AutoModel/model_backend.cpp"
            "${FLM_SOURCE_DIR}/common/AutoModel/modeling_gemma4e.cpp")
        target_include_directories(test_gemma4_frontend PRIVATE
            "${CMAKE_CURRENT_LIST_DIR}"
            "${FLM_SOURCE_DIR}/include"
            "${XRT_INCLUDE_DIR}"
            "${BOOST_INCLUDE_DIR}")
        target_compile_definitions(test_gemma4_frontend PRIVATE
            DEV_BUILD=1 __WINDOWS__ USEAVX2=1 DISABLE_ABI_CHECK=1
            _ENABLE_EXTENDED_ALIGNED_STORAGE WIN32_LEAN_AND_MEAN NOMINMAX)
        target_compile_options(test_gemma4_frontend PRIVATE
            $<$<CXX_COMPILER_ID:MSVC>:/wd4005 /wd4244>)
        target_link_directories(test_gemma4_frontend PRIVATE "${XRT_INCLUDE_DIR}/../lib")
        target_link_libraries(test_gemma4_frontend PRIVATE xrt_coreutil)
        add_test(NAME test_gemma4_frontend COMMAND test_gemma4_frontend)
    else()
        # LOUD AT CONFIGURE, AND RED AT CTEST -- the treatment C6 gave
        # test_gemma4_shape_plan and C7 gave test_gemma4_engine, arriving here
        # last. message(STATUS) alone is visible while configuring and
        # invisible afterwards: the ctest run that follows says "100% tests
        # passed" and nothing in it names the target that was never compiled.
        # A green run that covers less than it appears to is the same species
        # as the vacuous test, the self-confirming fixture and the stale .exe,
        # and this project has now been bitten by all four.
        message(WARNING "gemma4_rai: test_gemma4_frontend will NOT be built -- "
            "XRT_INCLUDE_DIR=${XRT_INCLUDE_DIR}, BOOST_INCLUDE_DIR=${BOOST_INCLUDE_DIR} "
            "(need XRT's xrt/xrt_bo.h and Boost's boost/any.hpp). "
            "ctest will report test_gemma4_frontend_WAS_NOT_BUILT as FAILED so "
            "this run cannot be mistaken for a complete one.")
        add_test(NAME test_gemma4_frontend_WAS_NOT_BUILT
            COMMAND "${CMAKE_COMMAND}" -E echo
                "test_gemma4_frontend WAS NOT BUILT: xrt/xrt_bo.h or "
                "boost/any.hpp was not found at configure time, so this ctest "
                "run does not cover it.")
        set_tests_properties(test_gemma4_frontend_WAS_NOT_BUILT PROPERTIES
            FAIL_REGULAR_EXPRESSION "WAS NOT BUILT")
    endif()
endif()

# --- test_gemma4_config (Task C1) ----------------------------------------
# Reads a Gemma4Config off a GGUF fixture. gemma4_rai_gguf.cpp is Task C1's
# own file; flm::rai::GgufFile (rai/gguf_file.cpp) is Task B1's. Neither
# includes a corelib header -- phi4_rai_gguf.cpp, the file this one is
# modelled on, does not either -- so this target needs no RYZENAI_CORELIB_
# INCLUDE_DIR and no fake_corelib. If the owning task's design ends up
# needing corelib here after all, add the find_path/link the same way
# test_gemma4_shape_plan does below.
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/test_gemma4_config.cpp")
    add_executable(test_gemma4_config
        test_gemma4_config.cpp
        "${FLM_SOURCE_DIR}/common/models/gemma4/rai/aie_next/gemma4_rai_gguf.cpp"
        "${FLM_SOURCE_DIR}/common/rai/gguf_file.cpp")
    target_include_directories(test_gemma4_config PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}" "${FLM_SOURCE_DIR}/include")
    add_test(NAME test_gemma4_config COMMAND test_gemma4_config)
endif()

# --- test_gemma4_derivations (Tasks C2-C4) -------------------------------
# Same sources as test_gemma4_config: FFN width, SWA layer period and cache
# ownership are all derivations read off the same gemma4_rai_gguf.cpp.
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/test_gemma4_derivations.cpp")
    add_executable(test_gemma4_derivations
        test_gemma4_derivations.cpp
        "${FLM_SOURCE_DIR}/common/models/gemma4/rai/aie_next/gemma4_rai_gguf.cpp"
        "${FLM_SOURCE_DIR}/common/rai/gguf_file.cpp")
    target_include_directories(test_gemma4_derivations PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}" "${FLM_SOURCE_DIR}/include")
    add_test(NAME test_gemma4_derivations COMMAND test_gemma4_derivations)
endif()

# --- test_gemma4_gguf (Task C5) ------------------------------------------
# The whole-package contract (Gemma4GgufPackage), same sources again.
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/test_gemma4_gguf.cpp")
    add_executable(test_gemma4_gguf
        test_gemma4_gguf.cpp
        "${FLM_SOURCE_DIR}/common/models/gemma4/rai/aie_next/gemma4_rai_gguf.cpp"
        "${FLM_SOURCE_DIR}/common/rai/gguf_file.cpp")
    target_include_directories(test_gemma4_gguf PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}" "${FLM_SOURCE_DIR}/include")
    add_test(NAME test_gemma4_gguf COMMAND test_gemma4_gguf)
endif()

# --- test_gemma4_real_gguf (Task R1) -------------------------------------
# THE ONLY TARGET HERE THAT READS A FILE GOOGLE SHIPPED. Everything above
# reads gemma4_gguf_fixture.hpp's synthetic file, and for four signed-off
# tasks that fixture and the readers written against it confirmed each other
# while both were transposed relative to every real conversion. This target
# is the answer to that; see test_gemma4_real_gguf.cpp's header comment.
#
# It is OPT-IN on two environment variables naming the real GGUFs, because
# they are 4.8 GB and 7.7 GB and will not exist on every machine. The skip
# convention is the tree's existing one for external artifacts --
# src/test/phi4_rai's test_real_corelib: the binary returns 77 and ctest
# prints "Skipped" on its own line, which is loud, unlike a target that
# silently is not built.
#
# This block IS an edit to a file whose header says later tasks should only
# add their own .cpp. That instruction exists so two agents working in
# parallel do not collide here; there was no block for a real-file target
# because until this task no real file could be opened at all.
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/test_gemma4_real_gguf.cpp")
    add_executable(test_gemma4_real_gguf
        test_gemma4_real_gguf.cpp
        "${FLM_SOURCE_DIR}/common/models/gemma4/rai/aie_next/gemma4_rai_gguf.cpp"
        "${FLM_SOURCE_DIR}/common/models/gemma4/rai/aie_next/gemma4_rai_host.cpp"
        "${FLM_SOURCE_DIR}/common/rai/gguf_file.cpp")
    target_include_directories(test_gemma4_real_gguf PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}" "${FLM_SOURCE_DIR}/include")
    add_test(NAME test_gemma4_real_gguf COMMAND test_gemma4_real_gguf)
    set_tests_properties(test_gemma4_real_gguf PROPERTIES SKIP_RETURN_CODE 77)
endif()

# --- test_gemma4_tokenizer (Task R5) -------------------------------------
# THE BINARY THAT TAKES A STEP OFF THE HARDWARE CHECKLIST. review-R4.md §5
# settled the doubled-BOS question on the host and recommended landing it as
# a test; this is it. It links the PRODUCTION Tokenizer class against the
# same tokenizers-cpp libraries the shipped binary links, and renders through
# the production minja chat template, so the answer is the shipped stack's
# answer and not a reimplementation of it.
#
# This is the only target here that needs a LIBRARY rather than a header.
# tokenizers-cpp is a git submodule plus a Rust staticlib, and building it
# from source here would add cargo to this suite's dependencies and minutes
# to its build, so the three libraries are located instead. They are produced
# by any build of src/CMakeLists.txt, under <build>/tokenizers-cpp:
#
#   tokenizers_c.lib          the Rust staticlib (cargo)          <dir>/
#   tokenizers_cpp.lib        the C++ binding                     <dir>/Release/
#   sentencepiece.lib         the other tokenizer backend         <dir>/sentencepiece/src/Release/
#
# Point FLM_TOKENIZERS_BUILD_DIR (cache or environment) at that directory if
# the hints below miss it.
#
# The header is IN-TREE (src/include/tokenizer/tokenizers_cpp.h), so the
# submodule does not need to be checked out.
#
# XRT and Boost are needed for a reason that has nothing to do with
# tokenizing: tokenizer.hpp includes typedef.hpp, which includes the FROZEN
# buffer.hpp, which includes device_runtime.hpp, which includes xrt/xrt_bo.h
# (and through xrt/detail/any.h, boost/any.hpp). Same probes as the frontend
# target above, including the fifth hint for a git worktree.
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/test_gemma4_tokenizer.cpp")
    find_path(XRT_INCLUDE_DIR NAMES xrt/xrt_bo.h
        HINTS "$ENV{XRT_INCLUDE_DIR}"
              "${CMAKE_CURRENT_LIST_DIR}/../../../../xrt_package/xrt/include"
              "${CMAKE_CURRENT_LIST_DIR}/../../../../../../xrt_package/xrt/include"
              "C:/dev/XRT/src/runtime_src/core/include")
    find_path(BOOST_INCLUDE_DIR NAMES boost/any.hpp
        HINTS "$ENV{CONDA_PREFIX}/Library/include"
              "$ENV{USERPROFILE}/anaconda3/Library/include")
    find_path(FLM_TOKENIZERS_BUILD_DIR NAMES tokenizers_c.lib
        HINTS "$ENV{FLM_TOKENIZERS_BUILD_DIR}"
              "${FLM_SOURCE_DIR}/build/tokenizers-cpp"
              "${FLM_SOURCE_DIR}/../build/tokenizers-cpp"
              # A sibling worktree: this port is developed in
              # <repo>/.worktrees/<name>, and the product has usually been
              # built in one of the others rather than in this one.
              "${CMAKE_CURRENT_LIST_DIR}/../../../../src/build/tokenizers-cpp")
    set(FLM_TOKENIZERS_CPP_LIB "${FLM_TOKENIZERS_BUILD_DIR}/Release/tokenizers_cpp.lib")
    set(FLM_SENTENCEPIECE_LIB "${FLM_TOKENIZERS_BUILD_DIR}/sentencepiece/src/Release/sentencepiece.lib")
    if(XRT_INCLUDE_DIR AND BOOST_INCLUDE_DIR AND FLM_TOKENIZERS_BUILD_DIR
       AND EXISTS "${FLM_TOKENIZERS_CPP_LIB}" AND EXISTS "${FLM_SENTENCEPIECE_LIB}")
        add_executable(test_gemma4_tokenizer
            test_gemma4_tokenizer.cpp
            "${FLM_SOURCE_DIR}/common/tokenizer/tokenizer.cpp")
        target_include_directories(test_gemma4_tokenizer PRIVATE
            "${CMAKE_CURRENT_LIST_DIR}"
            "${FLM_SOURCE_DIR}/include"
            "${FLM_SOURCE_DIR}/include/tokenizer"
            "${XRT_INCLUDE_DIR}"
            "${BOOST_INCLUDE_DIR}")
        target_compile_definitions(test_gemma4_tokenizer PRIVATE
            DEV_BUILD=1 __WINDOWS__ USEAVX2=1 DISABLE_ABI_CHECK=1
            _ENABLE_EXTENDED_ALIGNED_STORAGE WIN32_LEAN_AND_MEAN NOMINMAX)
        target_compile_options(test_gemma4_tokenizer PRIVATE
            $<$<CXX_COMPILER_ID:MSVC>:/wd4005 /wd4244 /bigobj>)
        target_link_directories(test_gemma4_tokenizer PRIVATE "${XRT_INCLUDE_DIR}/../lib")
        target_link_libraries(test_gemma4_tokenizer PRIVATE
            xrt_coreutil
            "${FLM_TOKENIZERS_CPP_LIB}"
            "${FLM_TOKENIZERS_BUILD_DIR}/tokenizers_c.lib"
            "${FLM_SENTENCEPIECE_LIB}"
            # What the Rust staticlib needs on Windows -- copied from
            # third_party/tokenizers-cpp/CMakeLists.txt's own list.
            ntdll wsock32 ws2_32 Bcrypt iphlpapi userenv psapi)
        add_test(NAME test_gemma4_tokenizer COMMAND test_gemma4_tokenizer)
        # Opt-in on the real packages, exactly like test_gemma4_real_gguf.
        set_tests_properties(test_gemma4_tokenizer PROPERTIES SKIP_RETURN_CODE 77)
    else()
        # LOUD AT CONFIGURE, AND RED AT CTEST -- the convention every block in
        # this file follows, and it matters more here than anywhere else:
        # silently dropping THIS target silently puts checklist step 0b back
        # on the hardware list, and nothing in a green ctest line would say
        # so. That is the exact failure -- a run that covers less than it
        # appears to -- this suite has been bitten by four times.
        message(WARNING "gemma4_rai: test_gemma4_tokenizer will NOT be built -- "
            "FLM_TOKENIZERS_BUILD_DIR=${FLM_TOKENIZERS_BUILD_DIR}, "
            "XRT_INCLUDE_DIR=${XRT_INCLUDE_DIR}, BOOST_INCLUDE_DIR=${BOOST_INCLUDE_DIR} "
            "(need tokenizers_c.lib, Release/tokenizers_cpp.lib and "
            "sentencepiece/src/Release/sentencepiece.lib, as produced by any "
            "build of src/CMakeLists.txt under <build>/tokenizers-cpp). "
            "Pass -DFLM_TOKENIZERS_BUILD_DIR=<dir> or set it in the "
            "environment. ctest will report test_gemma4_tokenizer_WAS_NOT_BUILT "
            "as FAILED so this run cannot be mistaken for a complete one.")
        add_test(NAME test_gemma4_tokenizer_WAS_NOT_BUILT
            COMMAND "${CMAKE_COMMAND}" -E echo
                "test_gemma4_tokenizer WAS NOT BUILT: the tokenizers-cpp "
                "libraries were not found at configure time, so the "
                "doubled-BOS question is NOT answered by this run and "
                "checklist step 0b is back on the hardware list. "
                "Re-configure with -DFLM_TOKENIZERS_BUILD_DIR=<dir>.")
        set_tests_properties(test_gemma4_tokenizer_WAS_NOT_BUILT PROPERTIES
            FAIL_REGULAR_EXPRESSION "WAS NOT BUILT")
    endif()
endif()

# --- test_gemma4_shape_plan (Task C6) ------------------------------------
# Builds corelib descriptors (flat_mha_bf16_desc, ssmlp) per geometry, so
# this one does need the corelib header and fake_corelib, the same way
# test_phi4_shape_plan does in src/test/phi4_rai/CMakeLists.txt.
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/test_gemma4_shape_plan.cpp")
    # Not REQUIRED -- see the comment on test_gemma4_frontend's probes above;
    # same reasoning applies to every target below.
    #
    # HINTS, though, and they are not cosmetic. This probe had none, so a
    # configure that did not pass -DRYZENAI_CORELIB_INCLUDE_DIR found nothing,
    # skipped this target, and then `ctest` reported "100% tests passed ... out
    # of 4" -- a green run over four targets that is indistinguishable, in a CI
    # line or a scrollback, from the green run over five. A review reproduced
    # exactly that. The pinned 0.5.0 header lives in a sibling worktree's
    # .deps, which is where the second and third hints point.
    find_path(RYZENAI_CORELIB_INCLUDE_DIR NAMES ryzenai/corelib.h
        HINTS "$ENV{RYZENAI_CORELIB_INCLUDE_DIR}"
              "${CMAKE_CURRENT_LIST_DIR}/../../../.deps/corelib-050/include")
    if(RYZENAI_CORELIB_INCLUDE_DIR)
        add_executable(test_gemma4_shape_plan
            test_gemma4_shape_plan.cpp
            "${CMAKE_CURRENT_LIST_DIR}/fake_corelib.cpp"
            "${FLM_SOURCE_DIR}/common/rai/corelib_api.cpp"
            "${FLM_SOURCE_DIR}/common/models/gemma4/rai/aie_next/gemma4_rai_gguf.cpp"
            "${FLM_SOURCE_DIR}/common/rai/gguf_file.cpp"
            "${FLM_SOURCE_DIR}/common/models/gemma4/rai/aie_next/gemma4_rai_shape_plan.cpp")
        target_include_directories(test_gemma4_shape_plan PRIVATE
            "${CMAKE_CURRENT_LIST_DIR}"
            "${FLM_SOURCE_DIR}/include"
            "${RYZENAI_CORELIB_INCLUDE_DIR}")
        target_compile_definitions(test_gemma4_shape_plan PRIVATE RYZENAI_CORELIB_STATIC=1)
        add_test(NAME test_gemma4_shape_plan COMMAND test_gemma4_shape_plan)
    else()
        # LOUD AT CONFIGURE, AND RED AT CTEST. message(STATUS) alone -- which
        # is what this branch used to be -- is visible when configuring and
        # invisible afterwards: the ctest run still says "100% tests passed"
        # and nothing in it names the target that was never compiled. A green
        # run that covers less than it appears to is the same species as the
        # vacuous test, the self-confirming fixture and the stale .exe, and
        # this project has been bitten by all four.
        #
        # So: WARNING rather than STATUS, plus a registered test that FAILS.
        # The test's command succeeds and prints why; FAIL_REGULAR_EXPRESSION
        # is what turns that into a ctest failure, so the explanation lands in
        # the failure output instead of having to be hunted for in the
        # configure log.
        #
        # This is deliberately NOT find_path(... REQUIRED): a configure that
        # only wants the dependency-free targets (test_gemma4_host and
        # friends) must still succeed, which is the convention the blocks
        # above established. It only refuses to let such a run LOOK complete.
        message(WARNING "gemma4_rai: test_gemma4_shape_plan will NOT be built -- "
            "RYZENAI_CORELIB_INCLUDE_DIR not found (need ryzenai/corelib.h). "
            "Pass -DRYZENAI_CORELIB_INCLUDE_DIR=<dir> or set it in the "
            "environment. ctest will report test_gemma4_shape_plan_WAS_NOT_BUILT "
            "as FAILED so this run cannot be mistaken for a complete one.")
        add_test(NAME test_gemma4_shape_plan_WAS_NOT_BUILT
            COMMAND "${CMAKE_COMMAND}" -E echo
                "test_gemma4_shape_plan WAS NOT BUILT: RYZENAI_CORELIB_INCLUDE_DIR "
                "was not found at configure time, so this ctest run does not "
                "cover it. Re-configure with -DRYZENAI_CORELIB_INCLUDE_DIR=<dir>.")
        set_tests_properties(test_gemma4_shape_plan_WAS_NOT_BUILT PROPERTIES
            FAIL_REGULAR_EXPRESSION "WAS NOT BUILT")
    endif()
endif()

# --- test_gemma4_engine (Tasks C7-C9) ------------------------------------
# The full engine: weights, the layer loop, teardown/poisoned(). Needs
# corelib, boost and XRT the same way test_phi4_engine does.
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/test_gemma4_engine.cpp")
    # HINTS, AND THEY ARE NOT COSMETIC -- this probe had none, exactly like
    # test_gemma4_shape_plan's did before C6 fixed it. A configure that did
    # not pass -DRYZENAI_CORELIB_INCLUDE_DIR found nothing, skipped the
    # target with a message(STATUS) nobody sees afterwards, and then `ctest`
    # reported "100% tests passed" over a set that did not include the
    # engine. A green run that covers less than it appears to is the failure
    # this project keeps paying for.
    find_path(RYZENAI_CORELIB_INCLUDE_DIR NAMES ryzenai/corelib.h
        HINTS "$ENV{RYZENAI_CORELIB_INCLUDE_DIR}"
              "${CMAKE_CURRENT_LIST_DIR}/../../../.deps/corelib-050/include")
    find_path(BOOST_INCLUDE_DIR NAMES boost/program_options.hpp
        HINTS "$ENV{CONDA_PREFIX}/Library/include"
              "$ENV{USERPROFILE}/anaconda3/Library/include")
    # The fifth hint is for a GIT WORKTREE. The fourth resolves to
    # <workspace>/xrt_package only when the checkout sits directly in the
    # workspace; a worktree under <repo>/.worktrees/<name> is two levels
    # deeper, so the same package needs two more "..". Without it this
    # target silently vanished in every worktree, which is where this port
    # is actually developed.
    find_path(XRT_INCLUDE_DIR NAMES xrt/xrt_bo.h
        HINTS "$ENV{XRT_INCLUDE_DIR}"
              "${CMAKE_CURRENT_LIST_DIR}/../../../../xrt_package/xrt/include"
              "${CMAKE_CURRENT_LIST_DIR}/../../../../../../xrt_package/xrt/include"
              "C:/dev/XRT/src/runtime_src/core/include")
    if(RYZENAI_CORELIB_INCLUDE_DIR AND BOOST_INCLUDE_DIR AND XRT_INCLUDE_DIR)
        add_executable(test_gemma4_engine
            test_gemma4_engine.cpp
            "${CMAKE_CURRENT_LIST_DIR}/fake_corelib.cpp"
            "${FLM_SOURCE_DIR}/common/rai/corelib_api.cpp"
            "${FLM_SOURCE_DIR}/common/rai/corelib_runtime.cpp"
            "${FLM_SOURCE_DIR}/common/rai/gguf_file.cpp"
            "${FLM_SOURCE_DIR}/common/rai/weight_cache.cpp"
            "${FLM_SOURCE_DIR}/common/models/gemma4/rai/aie_next/gemma4_rai_gguf.cpp"
            "${FLM_SOURCE_DIR}/common/models/gemma4/rai/aie_next/gemma4_rai_host.cpp"
            "${FLM_SOURCE_DIR}/common/models/gemma4/rai/aie_next/gemma4_rai_shape_plan.cpp"
            "${FLM_SOURCE_DIR}/common/models/gemma4/rai/aie_next/gemma4_rai.cpp")
        target_include_directories(test_gemma4_engine PRIVATE
            "${CMAKE_CURRENT_LIST_DIR}"
            "${FLM_SOURCE_DIR}/include"
            "${RYZENAI_CORELIB_INCLUDE_DIR}"
            "${BOOST_INCLUDE_DIR}"
            "${XRT_INCLUDE_DIR}")
        target_compile_definitions(test_gemma4_engine PRIVATE
            RYZENAI_CORELIB_STATIC=1 USEAVX2=1 DISABLE_ABI_CHECK=1
            _ENABLE_EXTENDED_ALIGNED_STORAGE WIN32_LEAN_AND_MEAN NOMINMAX
            # Task C9's structural teardown gate READS THE ENGINE'S SOURCE.
            # The order corelib requires -- every packed object released
            # before the Stream -- is a property of `Impl`'s MEMBER
            # DECLARATION ORDER, and the fault it prevents is intermittent
            # and raises nothing, so the only test with teeth is one that
            # reads the declarations themselves. The test FAILS, loudly, if
            # this path does not open; a structural gate that skips when it
            # cannot find its subject is worse than none.
            "FLM_GEMMA4_RAI_SOURCE=\"${FLM_SOURCE_DIR}/common/models/gemma4/rai/aie_next/gemma4_rai.cpp\"")
        target_compile_options(test_gemma4_engine PRIVATE
            $<$<CXX_COMPILER_ID:MSVC>:/wd4005 /wd4244>)
        target_link_directories(test_gemma4_engine PRIVATE "${XRT_INCLUDE_DIR}/../lib")
        target_link_libraries(test_gemma4_engine PRIVATE xrt_coreutil)
        add_test(NAME test_gemma4_engine COMMAND test_gemma4_engine)
    else()
        # LOUD AT CONFIGURE, AND RED AT CTEST -- the same treatment C6 gave
        # test_gemma4_shape_plan, and for the same reason: message(STATUS)
        # alone is invisible once the configure log has scrolled past, and
        # the ctest run that follows says "100% tests passed" without naming
        # the target that was never compiled.
        message(WARNING "gemma4_rai: test_gemma4_engine will NOT be built -- "
            "RYZENAI_CORELIB_INCLUDE_DIR=${RYZENAI_CORELIB_INCLUDE_DIR}, "
            "BOOST_INCLUDE_DIR=${BOOST_INCLUDE_DIR}, XRT_INCLUDE_DIR=${XRT_INCLUDE_DIR} "
            "(need ryzenai/corelib.h, boost/program_options.hpp, xrt/xrt_bo.h). "
            "ctest will report test_gemma4_engine_WAS_NOT_BUILT as FAILED so "
            "this run cannot be mistaken for a complete one.")
        add_test(NAME test_gemma4_engine_WAS_NOT_BUILT
            COMMAND "${CMAKE_COMMAND}" -E echo
                "test_gemma4_engine WAS NOT BUILT: one of ryzenai/corelib.h, "
                "boost/program_options.hpp or xrt/xrt_bo.h was not found at "
                "configure time, so this ctest run does not cover it.")
        set_tests_properties(test_gemma4_engine_WAS_NOT_BUILT PROPERTIES
            FAIL_REGULAR_EXPRESSION "WAS NOT BUILT")
    endif()
endif()

# --- test_gemma4_backend (Task C10) --------------------------------------
# The backend, its traits, and registration under family gemma4e. Same
# dependency shape as test_gemma4_engine, plus the backend's own .cpp.
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/test_gemma4_backend.cpp")
    # HINTS, AND THEY ARE NOT COSMETIC -- this block was written
    # speculatively, before C6 and C7 had found what a hintless probe costs.
    # The corelib probe had NO hints at all, so a configure that did not pass
    # -DRYZENAI_CORELIB_INCLUDE_DIR found nothing and dropped this target with
    # a message(STATUS) nobody sees afterwards; and the XRT probe was missing
    # the fifth hint, the one for a GIT WORKTREE (the fourth resolves to
    # <workspace>/xrt_package only when the checkout sits directly in the
    # workspace; a worktree under <repo>/.worktrees/<name> is two levels
    # deeper). Either way `ctest` then reports "100% tests passed" over a set
    # that does not include the backend. Fixed here the same way C6 fixed
    # test_gemma4_shape_plan and C7 fixed test_gemma4_engine.
    find_path(RYZENAI_CORELIB_INCLUDE_DIR NAMES ryzenai/corelib.h
        HINTS "$ENV{RYZENAI_CORELIB_INCLUDE_DIR}"
              "${CMAKE_CURRENT_LIST_DIR}/../../../.deps/corelib-050/include")
    find_path(BOOST_INCLUDE_DIR NAMES boost/program_options.hpp
        HINTS "$ENV{CONDA_PREFIX}/Library/include"
              "$ENV{USERPROFILE}/anaconda3/Library/include")
    find_path(XRT_INCLUDE_DIR NAMES xrt/xrt_bo.h
        HINTS "$ENV{XRT_INCLUDE_DIR}"
              "${CMAKE_CURRENT_LIST_DIR}/../../../../xrt_package/xrt/include"
              "${CMAKE_CURRENT_LIST_DIR}/../../../../../../xrt_package/xrt/include"
              "C:/dev/XRT/src/runtime_src/core/include")
    if(RYZENAI_CORELIB_INCLUDE_DIR AND BOOST_INCLUDE_DIR AND XRT_INCLUDE_DIR)
        add_executable(test_gemma4_backend
            test_gemma4_backend.cpp
            "${CMAKE_CURRENT_LIST_DIR}/fake_corelib.cpp"
            # The registry itself, so registration, traits lookup and
            # resolve_backend_id all run their production code. The one thing
            # this target does NOT link is builtin_backends.cpp: that file
            # names every engine type and would drag in the ~20 prebuilt
            # libraries under src/lib/<runtime>. The test defines its own
            # empty register_builtin_backends instead -- the same trick
            # src/test/phi4_rai/test_phi4_frontend.cpp uses.
            "${FLM_SOURCE_DIR}/common/AutoModel/model_backend.cpp"
            "${FLM_SOURCE_DIR}/common/rai/corelib_api.cpp"
            "${FLM_SOURCE_DIR}/common/rai/corelib_runtime.cpp"
            "${FLM_SOURCE_DIR}/common/rai/gguf_file.cpp"
            "${FLM_SOURCE_DIR}/common/rai/weight_cache.cpp"
            "${FLM_SOURCE_DIR}/common/models/gemma4/rai/aie_next/gemma4_rai_gguf.cpp"
            "${FLM_SOURCE_DIR}/common/models/gemma4/rai/aie_next/gemma4_rai_host.cpp"
            "${FLM_SOURCE_DIR}/common/models/gemma4/rai/aie_next/gemma4_rai_shape_plan.cpp"
            "${FLM_SOURCE_DIR}/common/models/gemma4/rai/aie_next/gemma4_rai.cpp"
            "${FLM_SOURCE_DIR}/common/models/gemma4/rai/aie_next/gemma4_rai_backend.cpp")
        target_include_directories(test_gemma4_backend PRIVATE
            "${CMAKE_CURRENT_LIST_DIR}"
            "${FLM_SOURCE_DIR}/include"
            "${RYZENAI_CORELIB_INCLUDE_DIR}"
            "${BOOST_INCLUDE_DIR}"
            "${XRT_INCLUDE_DIR}")
        target_compile_definitions(test_gemma4_backend PRIVATE
            RYZENAI_CORELIB_STATIC=1 USEAVX2=1 DISABLE_ABI_CHECK=1
            _ENABLE_EXTENDED_ALIGNED_STORAGE WIN32_LEAN_AND_MEAN NOMINMAX
            # The registration this target asserts is the one builtin_backends
            # .cpp makes under `#if defined(FLM_ENABLE_RAI)`, so the guard has
            # to be on here too -- otherwise the assertion is about a branch
            # the shipping build does not take.
            FLM_ENABLE_RAI=1
            # Turns flm::file_access's observer on. Without it ObserveOpen is
            # an empty inline and "nothing was opened before the rejection"
            # is vacuously true -- the exact shape of vacuous test this
            # project has been bitten by.
            FLM_CORELIB_TESTING=1
            # The BACKEND's teardown gate READS ITS OWN SOURCE, the same way
            # C9's engine gate reads gemma4_rai.cpp. "The runtime shared_ptr
            # is declared before the engine member" is what makes the engine
            # release before the runtime, it is invisible at runtime (the
            # fault corelib raises is intermittent and raises nothing here),
            # and a later edit could reorder two member declarations without
            # a single test noticing. The test FAILS, loudly, if this path
            # does not open.
            "FLM_GEMMA4_RAI_BACKEND_SOURCE=\"${FLM_SOURCE_DIR}/common/models/gemma4/rai/aie_next/gemma4_rai_backend.cpp\""
            # C-2's two gates read their subjects the same way. Neither
            # function can be LINKED here: get_auto_model includes every
            # frontend in the tree, and ensure_model_loaded is the whole
            # server. What this binary can still say is that the server
            # refuses an unknown model name before it resolves it, and that
            # get_auto_model no longer picks a frontend from a literal tag.
            # Both tests FAIL, loudly, if these paths do not open.
            "FLM_REST_HANDLER_SOURCE=\"${FLM_SOURCE_DIR}/server/rest_handler.cpp\""
            "FLM_ALL_MODELS_SOURCE=\"${FLM_SOURCE_DIR}/include/AutoModel/all_models.hpp\""
            # The SHIPPED catalog, for the two tripwires D1 left: both are
            # about fallback_model() becoming reachable for a family that now
            # has a rai backend, and neither can be asked of a hand-written
            # catalog -- the whole question is what the real model_list.json
            # resolves to on aie_next.
            "FLM_TEST_MODEL_LIST_PATH=\"${FLM_SOURCE_DIR}/model_list.json\"")
        target_compile_options(test_gemma4_backend PRIVATE
            $<$<CXX_COMPILER_ID:MSVC>:/wd4005 /wd4244>)
        target_link_directories(test_gemma4_backend PRIVATE "${XRT_INCLUDE_DIR}/../lib")
        target_link_libraries(test_gemma4_backend PRIVATE xrt_coreutil)
        add_test(NAME test_gemma4_backend COMMAND test_gemma4_backend)
    else()
        # LOUD AT CONFIGURE, AND RED AT CTEST -- this block shipped with a
        # bare message(STATUS), which is visible while configuring and
        # invisible afterwards: the ctest run that follows says "100% tests
        # passed" and nothing in it names the target that was never compiled.
        # Same treatment as every other block above, for the same reason.
        message(WARNING "gemma4_rai: test_gemma4_backend will NOT be built -- "
            "RYZENAI_CORELIB_INCLUDE_DIR=${RYZENAI_CORELIB_INCLUDE_DIR}, "
            "BOOST_INCLUDE_DIR=${BOOST_INCLUDE_DIR}, XRT_INCLUDE_DIR=${XRT_INCLUDE_DIR} "
            "(need ryzenai/corelib.h, boost/program_options.hpp, xrt/xrt_bo.h). "
            "ctest will report test_gemma4_backend_WAS_NOT_BUILT as FAILED so "
            "this run cannot be mistaken for a complete one.")
        add_test(NAME test_gemma4_backend_WAS_NOT_BUILT
            COMMAND "${CMAKE_COMMAND}" -E echo
                "test_gemma4_backend WAS NOT BUILT: one of ryzenai/corelib.h, "
                "boost/program_options.hpp or xrt/xrt_bo.h was not found at "
                "configure time, so this ctest run does not cover it.")
        set_tests_properties(test_gemma4_backend_WAS_NOT_BUILT PROPERTIES
            FAIL_REGULAR_EXPRESSION "WAS NOT BUILT")
    endif()
endif()

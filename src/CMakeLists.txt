cmake_minimum_required(VERSION 3.22)
project(flm LANGUAGES CXX VERSION 0.1.0)

if(POLICY CMP0207)
    cmake_policy(SET CMP0207 NEW)
endif()

if(NOT DEFINED FLM_VERSION)
    message(FATAL_ERROR "FLM_VERSION must be specified externally. Use -DFLM_VERSION=<version> when running cmake.")
endif()

if(NOT DEFINED NPU_VERSION)
    message(FATAL_ERROR "NPU_VERSION must be specified externally. Use -DNPU_VERSION=<version> when running cmake.")
endif()

# Set build type to Release
set(CMAKE_BUILD_TYPE Release)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure CMake to stop at first error
set(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION ON)
set(CMAKE_ERROR_DEPRECATED ON)


# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/)


# Force output directories to be absolute
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

# ———————————————————————————————————————————————
# HRX amdxdna runtime for NPU dispatch (Linux + Windows).
# ———————————————————————————————————————————————
find_package(hrx CONFIG REQUIRED)

# ———————————————————————————————————————————————
# Add tokenizers-cpp subproject
# ———————————————————————————————————————————————
# Ensure C++17 is available for tokenizers-cpp
if(CMAKE_CXX_STANDARD LESS 17)
    set(CMAKE_CXX_STANDARD 17)
endif()

add_subdirectory(${CMAKE_SOURCE_DIR}/../third_party/tokenizers-cpp
                 ${CMAKE_BINARY_DIR}/tokenizers-cpp
                 EXCLUDE_FROM_ALL)

# ———————————————————————————————————————————————
# Gather your sources
# ———————————————————————————————————————————————
file(GLOB SOURCES "src/*.cpp" "runner/*.cpp" "common/*.cpp" "common/*/*.cpp" "server/*.cpp" "pull/*.cpp" )
file(GLOB HEADERS "include/*.hpp" "runner/*.hpp" "common/*.hpp" "common/*/*.hpp" "server/*.hpp" "pull/*.hpp")

# Exclude files that depend on missing libraries for Linux
if(NOT WIN32)
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_gemma3\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_gemma3_image\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_gemma3_text\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_gpt_oss\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_lfm2\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_phi4\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_qwen2\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_qwen3\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_qwen3vl\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_qwen3vl_image\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_whisper\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_whisper_audio\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_gemma_embedding\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*auto_embedding_model\\.cpp$")

    set_source_files_properties(
        ${CMAKE_SOURCE_DIR}/common/image_process_utils/imageprocAVX512.cpp
        ${CMAKE_SOURCE_DIR}/common/audio_process_utils/audioprocAVX512.cpp
        PROPERTIES
        COMPILE_OPTIONS "-mavx512f;-mavx512dq;-mavx512vl;-mavx512bw;-mfma"
    )

    # Define a macro to indicate limited model support on Linux
    # add_compile_definitions(FASTFLOWLM_LINUX_LIMITED_MODELS=1)
endif()


add_executable(flm ${SOURCES} ${HEADERS})

if(WIN32)
    if(VCPKG_TOOLCHAIN)
        # A vcpkg toolchain is active (e.g. the rocm-npu-staging dev.py build or
        # a local CMakePresets build): the vcpkg tree ships CMake package configs,
        # so resolve the native deps in CONFIG mode. find_package auto-detects the
        # correct versioned import-lib names, so this works regardless of the
        # installed Boost/FFTW versions.
        find_package(Boost CONFIG REQUIRED COMPONENTS program_options)
        find_package(CURL CONFIG REQUIRED)
        find_package(FFMPEG REQUIRED)
        find_package(FFTW3 CONFIG REQUIRED)
        find_package(FFTW3f CONFIG REQUIRED)
        find_package(FFTW3l CONFIG REQUIRED)
    endif()
    # Otherwise (the bare self-hosted CI runner) there is no vcpkg toolchain and
    # no Boost in vcpkg: Boost is a standalone b2 build under C:/dev/boost_1_88_0
    # and curl/ffmpeg/fftw are linked by raw name from the vcpkg lib dir. That
    # path (mirroring main, which builds green on this runner) is wired via the
    # include/link dirs and raw library names guarded by NOT VCPKG_TOOLCHAIN below.
else()
    find_package(Boost CONFIG REQUIRED COMPONENTS program_options)
    # The Linux build environment installs these via apt, which provides the
    # CMake FindCURL module and pkg-config files rather than CMake package
    # configs. Resolve CURL via the module and FFmpeg/FFTW via pkg-config.
    find_package(CURL REQUIRED)
    find_package(Threads REQUIRED)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FFTW3 REQUIRED IMPORTED_TARGET fftw3)
    pkg_check_modules(FFTW3F REQUIRED IMPORTED_TARGET fftw3f)
    pkg_check_modules(FFTW3L REQUIRED IMPORTED_TARGET fftw3l)
    pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
        libavformat libavcodec libavutil libswscale libswresample)
    pkg_check_modules(readline REQUIRED IMPORTED_TARGET readline)
    pkg_check_modules(ncurses REQUIRED IMPORTED_TARGET ncursesw)
endif()

# ———————————————————————————————————————————————
# Library specific settings
# ———————————————————————————————————————————————
target_include_directories(flm PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/runner
    ${CMAKE_SOURCE_DIR}/server
    ${CMAKE_SOURCE_DIR}/pull
    ${FFMPEG_INCLUDE_DIRS}
)

if(WIN32 AND NOT VCPKG_TOOLCHAIN)
    # Bare CI runner: standalone Boost + vcpkg headers by absolute path.
    target_include_directories(flm PUBLIC
        C:/dev/boost_1_88_0
        C:/dev/vcpkg/installed/x64-windows/include/
    )
endif()

target_compile_definitions(flm PUBLIC
    DISABLE_ABI_CHECK=1
    CMAKE_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}"
    CMAKE_XCLBIN_PREFIX="${CMAKE_XCLBIN_PREFIX}"
    __FLM_VERSION__=\"${FLM_VERSION}\"
    __NPU_VERSION__=\"${NPU_VERSION}\"
)

if(WIN32)
    target_compile_definitions(flm PUBLIC
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        # Handle legacy stdio functions
        _CRT_SECURE_NO_WARNINGS
        _CRT_NONSTDC_NO_DEPRECATE
        __WINDOWS__
    )
endif()

if(WIN32 AND NOT VCPKG_TOOLCHAIN)
    # Bare CI runner: statically link the standalone Boost + curl (matches main).
    target_compile_definitions(flm PUBLIC
        CURL_STATICLIB
        BOOST_ALL_NO_LIB
        BOOST_ALL_STATIC_LINK
    )
endif()

target_link_directories(flm PUBLIC
    ${CMAKE_SOURCE_DIR}/lib
    ${FFMPEG_LIBRARY_DIRS}
)

if(WIN32 AND NOT VCPKG_TOOLCHAIN)
    # Bare CI runner: standalone Boost stage libs + vcpkg import libs.
    target_link_directories(flm PUBLIC
        C:/dev/boost_1_88_0/stage/lib
        C:/dev/vcpkg/installed/x64-windows/lib
    )
endif()

# Link static libraries first
if(MSVC)
    target_link_libraries(flm PUBLIC ${STATIC_LIBS})
endif()

# Link your custom libraries (these may still be DLLs if no static versions available)
target_link_libraries(flm PUBLIC
    q4_npu_eXpress
    llama_npu
    qwen2_npu
    qwen2vl_npu
    qwen3_npu
    qwen3vl_npu
    qwen3_5vl_npu
    qwen3_5_omni_npu
    qwen3_6_moe_npu
    gemma_npu
    gemma_text_npu
    gemma4e_npu
    gpt_oss_npu
    whisper_npu
    gemma_embedding
    lfm2_npu
    phi4_npu
    nanbeige_npu
    dequant
    gemm
    lm_head
    mha
)

# Link the HRX runtime after the detail model libraries above so their libhrx
# references resolve. (Windows: hrx.lib import lib for hrx.dll; Linux: libhrx.so.)
target_link_libraries(flm PUBLIC hrx::hrx)
if(NOT WIN32)
    set_target_properties(flm PROPERTIES
        BUILD_RPATH "$<TARGET_FILE_DIR:hrx::hrx>;${CMAKE_SOURCE_DIR}/lib")
endif()

if(WIN32 AND VCPKG_TOOLCHAIN)
    # Local/managed vcpkg: link the imported targets from the CONFIG packages
    # found above (versioned import-lib names resolved automatically).
    target_link_libraries(flm PUBLIC
        ${FFMPEG_LIBRARIES}
        Boost::program_options
        CURL::libcurl
        FFTW3::fftw3
        FFTW3::fftw3f
        FFTW3::fftw3l
    )
elseif(WIN32)
    # Bare CI runner: link the native deps by raw name from the standalone Boost
    # + vcpkg lib dirs wired above (mirrors main; no CMake package configs there).
    target_link_libraries(flm PUBLIC
        avformat
        avcodec
        avutil
        swscale
        swresample
        libcurl
        libboost_program_options-vc143-mt-x64-1_88
        libfftw3-3
        libfftw3f-3
        libfftw3l-3
    )
else()
    target_link_libraries(flm PUBLIC
        ${FFMPEG_LIBRARIES}
        Boost::program_options
        CURL::libcurl
        PkgConfig::FFTW3
        PkgConfig::FFTW3F
        PkgConfig::FFTW3L
    )
endif()

if(NOT WIN32)
    target_compile_options(flm PUBLIC -mavx -mavx2)

    target_compile_definitions(flm PUBLIC FASTFLOWLM_USE_READLINE=1)
    target_link_libraries(flm PUBLIC PkgConfig::readline PkgConfig::ncurses)

    # Link with dl library for dlopen() support (used for preloading bundled libraries)
    target_link_libraries(flm PUBLIC dl)
endif()

# ———————————————————————————————————————————————
# Link tokenizers-cpp libraries
# ———————————————————————————————————————————————
# The tokenizers-cpp subproject provides these targets:
# - tokenizers_cpp: main tokenizers C++ library
# - tokenizers_c: tokenizers C FFI bindings  
# - sentencepiece: sentencepiece tokenizer library
target_link_libraries(flm PRIVATE
    tokenizers_cpp
)

if(WIN32)
    target_link_libraries(flm PRIVATE
        ntdll wsock32 ws2_32 Bcrypt
        iphlpapi userenv psapi
        crypt32 secur32 advapi32 normaliz wldap32
    )
else()
    target_link_libraries(flm PRIVATE Threads::Threads)
endif()


# ———————————————————————————————————————————————
# Copy the build flm.exe into the out directory (for local dev)
# ———————————————————————————————————————————————
if(WIN32)
    add_custom_command(TARGET flm POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:flm>
            ${CMAKE_SOURCE_DIR}/out/flm.exe
    )
endif()

# Default install location for model_list.json / xclbins (matches the app's
# relocatable "<exe_dir>/../share/flm" lookup). Overridden to the prefix root for
# the flat portable layout below.
set(FLM_SHARE_DESTINATION "share/flm")

# Add a custom target to check for remaining DLL dependencies
if(WIN32)
    add_custom_target(check_dependencies ALL
        COMMAND ${CMAKE_COMMAND} -E echo "Checking for DLL dependencies..."
        COMMAND dumpbin /dependents $<TARGET_FILE:flm> | findstr /i ".dll"
        COMMENT "Checking for remaining DLL dependencies"
        DEPENDS flm
    )
else()
    if(NOT CMAKE_INSTALL_LIBDIR)
        set(CMAKE_INSTALL_LIBDIR "lib")
    endif()

    # ————————————————————————————————————————————————————————————
    # Install layout. Portable = flat, self-contained tree rooted at the install
    # prefix (consumed by the portable tarball workflow); otherwise an FHS layout
    # (bin/, lib/, share/) consumed by the .deb packaging.
    # ————————————————————————————————————————————————————————————
    if(FLM_PORTABLE_BUILD)
        set(FLM_BIN_DESTINATION ".")
        set(FLM_ENGINE_LIB_DESTINATION "${CMAKE_INSTALL_LIBDIR}")
        set(FLM_SHARE_DESTINATION ".")
        set(FLM_FLM_INSTALL_RPATH "$ORIGIN/${CMAKE_INSTALL_LIBDIR}")
        # Engine libs sit next to libhrx in <root>/lib.
        set(FLM_ENGINE_INSTALL_RPATH "$ORIGIN")
    else()
        set(FLM_BIN_DESTINATION "bin")
        set(FLM_ENGINE_LIB_DESTINATION "${CMAKE_INSTALL_LIBDIR}/flm")
        set(FLM_SHARE_DESTINATION "share/flm")
        set(FLM_FLM_INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/flm;$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
        # Engine libs sit in <prefix>/lib/flm; libhrx is bundled one level up in <prefix>/lib.
        set(FLM_ENGINE_INSTALL_RPATH "$ORIGIN:$ORIGIN/..")
    endif()

    file(GLOB so_libs "${CMAKE_SOURCE_DIR}/lib/*.so*")
    install(FILES ${so_libs} DESTINATION "${FLM_ENGINE_LIB_DESTINATION}")
    set_target_properties(flm PROPERTIES INSTALL_RPATH "${FLM_FLM_INSTALL_RPATH}")

    # The prebuilt engine .so ship with an absolute build-machine RUNPATH and a
    # libhrx.so.0 NEEDED entry. Rewrite the RUNPATH to a relocatable $ORIGIN-based
    # one so libhrx resolves at runtime and, crucially, so dpkg-shlibdeps can locate
    # it during .deb packaging. Only the engine libs (by name) are patched, so
    # libhrx / bundled system libs that may share the dir in portable mode are left
    # untouched.
    find_program(PATCHELF_EXECUTABLE patchelf REQUIRED)
    set(_flm_engine_names "")
    set(_flm_engine_post_exclude "")
    foreach(_flm_src ${so_libs})
        get_filename_component(_flm_name "${_flm_src}" NAME)
        list(APPEND _flm_engine_names "${_flm_name}")
        string(REPLACE "." "\\." _flm_name_re "${_flm_name}")
        list(APPEND _flm_engine_post_exclude ".*/${_flm_name_re}$")
    endforeach()
    install(CODE "
        set(_flm_engine_names \"${_flm_engine_names}\")
        foreach(_flm_name \${_flm_engine_names})
            set(_flm_lib \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${FLM_ENGINE_LIB_DESTINATION}/\${_flm_name}\")
            if(EXISTS \"\${_flm_lib}\" AND NOT IS_SYMLINK \"\${_flm_lib}\")
                execute_process(
                    COMMAND \"${PATCHELF_EXECUTABLE}\" --set-rpath \"${FLM_ENGINE_INSTALL_RPATH}\" \"\${_flm_lib}\"
                    RESULT_VARIABLE _flm_patchelf_rc)
                if(NOT _flm_patchelf_rc EQUAL 0)
                    message(FATAL_ERROR \"patchelf --set-rpath failed for \${_flm_lib}\")
                endif()
            endif()
        endforeach()
    ")

    # Portable build ships a small wrapper as `flm`; the packaging step renames the
    # real binary to `flm-real` and this wrapper to `flm`.
    if(FLM_PORTABLE_BUILD)
        configure_file(
            "${CMAKE_SOURCE_DIR}/flm-wrapper.sh.in"
            "${CMAKE_BINARY_DIR}/flm-wrapper.sh"
            @ONLY)
        install(PROGRAMS "${CMAKE_BINARY_DIR}/flm-wrapper.sh" DESTINATION "${FLM_SHARE_DESTINATION}")
    endif()
endif()
if(WIN32)
    install(TARGETS flm
        RUNTIME_DEPENDENCY_SET fastflowlm_runtime_dependencies
        RUNTIME DESTINATION bin)
    install(RUNTIME_DEPENDENCY_SET fastflowlm_runtime_dependencies
        DESTINATION bin
        DIRECTORIES
            "${CMAKE_SOURCE_DIR}/lib"
            "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin"
            "$<TARGET_FILE_DIR:hrx::hrx>"
        PRE_EXCLUDE_REGEXES "api-ms-.*" "ext-ms-.*"
            # Optional Windows feature-on-demand / attestation / lockdown DLLs
            # that are not present as files on all hosts, so the resolver would
            # report them unresolved. They are OS components, never bundled.
            # Patterns are case-insensitive because PE import names are lowercase.
            "[Aa][Zz][Uu][Rr][Ee][Aa][Tt][Tt][Ee][Ss][Tt].*"
            "[Hh][Vv][Ss][Ii][Ff][Ii][Ll][Ee][Tt][Rr][Uu][Ss][Tt].*"
            "[Pp][Dd][Mm][Uu][Tt][Ii][Ll][Ii][Tt][Ii][Ee][Ss].*"
            "[Ww][Pp][Aa][Xx][Hh][Oo][Ll][Dd][Ee][Rr].*"
        POST_EXCLUDE_REGEXES ".*[Ww]indows[/\\\\][Ss]ystem32[/\\\\].*")
else()
    install(TARGETS flm
        RUNTIME_DEPENDENCY_SET fastflowlm_runtime_dependencies
        RUNTIME DESTINATION "${FLM_BIN_DESTINATION}")
    install(RUNTIME_DEPENDENCY_SET fastflowlm_runtime_dependencies
        DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        DIRECTORIES
            "${CMAKE_SOURCE_DIR}/lib"
            "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib"
            "$<TARGET_FILE_DIR:hrx::hrx>"
        PRE_EXCLUDE_REGEXES
            "linux-vdso\\.so.*"
            "ld-linux.*"
            "lib(c|dl|m|pthread|rt)\\.so.*"
        # The engine .so are installed (and RUNPATH-patched) explicitly into
        # FLM_ENGINE_LIB_DESTINATION above. Exclude them here so the dependency
        # resolver does not additionally drop unpatched copies (which retain the
        # absolute build-machine RUNPATH) into ${CMAKE_INSTALL_LIBDIR}. Their
        # transitive dependencies (libhrx, libgomp, ...) are still resolved and
        # bundled because POST_EXCLUDE only skips installation, not traversal.
        POST_EXCLUDE_REGEXES ${_flm_engine_post_exclude})
endif()

install(FILES model_list.json DESTINATION "${FLM_SHARE_DESTINATION}")

# xclbins, which are loaded by shared libraries need to be in location
# relative to the executable, so we install them relative to the binary.
install(DIRECTORY xclbins DESTINATION "${FLM_SHARE_DESTINATION}")


# Check if it is in Windows Subsystem for Linux (WSL)
ifeq ($(shell uname -a | grep -i WSL),)
	WSL := 0
else
	WSL := 1
endif

# get current directory name
CURRENT_DIR := $(notdir $(CURDIR))

BUILD_DIR := ../../build/test/$(CURRENT_DIR)

# ————————————————————————————————————————————————————————————————
# NPU runtime backend selection (mirrors the CMake FLM_USE_HRX flag).
#   FLM_USE_HRX=0 -> XRT (default)   FLM_USE_HRX=1 -> HRX
# Prebuilt engine libs are consumed from the matching per-backend subdir.
# ————————————————————————————————————————————————————————————————
FLM_USE_HRX ?= 0

ifeq ($(WSL), 0)
# Linux build environment
# Use g++-13 directly without CMake

CXX := g++-13
CXX_FLAGS := -std=c++20 -fPIC -Wall -DUSEAVX2=1
CXX_FLAGS += -mavx2 -mfma -march=native -ffast-math
CXX_FLAGS += -fmax-errors=1
CXX_FLAGS += -I../../include
CXX_FLAGS += -MMD -MP
CXX_FLAGS += -DDEV_BUILD
CXX_FLAGS += -fopenmp
CXX_FLAGS += -DCMAKE_INSTALL_PREFIX="\"/opt/fastflowlm\""
CXX_FLAGS += -DCMAKE_XCLBIN_PREFIX="\"/opt/fastflowlm/share/flm/xclbins\""
#NOTE: TODO: FIXME: Either deprecate makefile, or keep the parameter sync with ../CMAKELists.txt, otherwise it is error-prone
CXX_FLAGS += -D__FLM_VERSION__="\"0.9.34\""
CXX_FLAGS += -D__NPU_VERSION__="\"32.0.203.304\""
CXX_FLAGS += -DDISABLE_ABI_CHECK=1

ifeq ($(FLM_USE_HRX),1)
# --- HRX backend ---
# Consumed from the HRX dev package built from the public FastFlow SDK by
# hrx-integration/provision-hrx-from-fastflow-sdk.sh (pin:
# hrx-integration/fastflow-sdk.env). The package layout is the public one:
# include/hrx/*.h, a single lib/libhrx.so, and lib/cmake/hrx. Source the
# package's env.sh (exports HRX_HOME) or pass HRX_HOME=<package-prefix> on the
# make command line; otherwise this auto-locates the staged package.
LIB_DIR := ../../lib/hrx
HRX_MK_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO_ROOT  := $(abspath $(HRX_MK_DIR)/../..)
HRX_RELEASE_DIR ?= $(REPO_ROOT)/hrx-integration/.hrx-release
# Package prefix (contains include/, lib/, lib/cmake/hrx). Prefer HRX_HOME (set
# by the package env.sh / command line); otherwise auto-locate the staged
# package by its libhrx.so (top-level or one directory deep).
ifeq ($(origin HRX_HOME),undefined)
HRX_HOME := $(patsubst %/lib/libhrx.so,%,$(firstword $(wildcard \
              $(HRX_RELEASE_DIR)/lib/libhrx.so \
              $(HRX_RELEASE_DIR)/*/lib/libhrx.so)))
endif
ifeq ($(wildcard $(HRX_HOME)/lib/libhrx.so),)
$(error HRX public package not found (HRX_HOME="$(HRX_HOME)"). Run hrx-integration/provision-hrx-from-fastflow-sdk.sh (or `source <package>/env.sh`, or pass HRX_HOME=<package-prefix>) before building.)
endif
# Public headers install under include/hrx (hrx_amdxdna.h, ...); include/ is
# added too so both spellings resolve.
HRX_INC := -I$(HRX_HOME)/include/hrx -I$(HRX_HOME)/include
HRX_LIBS := -L$(HRX_HOME)/lib -lhrx
HRX_RPATH := -Wl,-rpath,$(HRX_HOME)/lib -Wl,-rpath,$(HRX_HOME)/lib/rocm_sysdeps/lib
CXX_FLAGS += -DFLM_USE_HRX=1 $(HRX_INC)
RUNTIME_LDFLAGS := $(HRX_RPATH) $(HRX_LIBS)
else
# --- XRT backend (default) ---
LIB_DIR := ../../lib/xrt
XRT_PREFIX ?= /opt/xilinx/xrt
CXX_FLAGS += -I$(XRT_PREFIX)/include
RUNTIME_LDFLAGS := -L$(XRT_PREFIX)/lib -Wl,-rpath,$(XRT_PREFIX)/lib -lxrt_coreutil -laiebu
endif

LDFLAGS += -lboost_program_options -lboost_filesystem
LDFLAGS += -L$(LIB_DIR)
# Resolve the prebuilt engine .so (and bundled aiebu) at run time.
LDFLAGS += -Wl,-rpath,$(abspath $(LIB_DIR))
LDFLAGS += -L../../build/tokenizers-cpp
LDFLAGS += -L../../build/tokenizers-cpp/sentencepiece/src
LDFLAGS += -ltokenizers_cpp -ltokenizers_c -lsentencepiece
LDFLAGS += $(RUNTIME_LDFLAGS)
DEPENDENCY_LDFLAGS += -lmha -ldequant -lgemm -llm_head -lq4_npu_eXpress


SOURCES += ../../common/utils.cpp

else

# WSL build environment
# Use CMake to invoke the Visual Studio
PWSH := powershell.exe

endif 

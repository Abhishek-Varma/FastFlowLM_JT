
# Check if it is in Windows Subsystem for Linux (WSL)
ifeq ($(shell uname -a | grep -i WSL),)
	WSL := 0
else
	WSL := 1
endif

# get current directory name
CURRENT_DIR := $(notdir $(CURDIR))

BUILD_DIR := ../../build/test/$(CURRENT_DIR)
LIB_DIR := ../../lib

ifeq ($(WSL), 0)
# Linux build environment
# Use g++-13 directly without CMake

CXX := g++-13

# --- HRX backend (replaces XRT + aiebu) ---
HRX_DIR   ?= /home/nod/jorn_repro/hrx
HRX_BUILD ?= $(HRX_DIR)/build/cmake
HRX_INC := -I$(HRX_DIR)/libhrx/include -I$(HRX_DIR)/runtime/src \
           -I$(HRX_BUILD)/runtime/src -I$(HRX_BUILD)/_deps/flatcc-src/include
HRX_LIBS := $(HRX_BUILD)/libhrx/src/libhrx/libhrx.so $(HRX_BUILD)/libflatcc_runtime.a
HRX_RPATH := -Wl,-rpath,$(HRX_BUILD)/libhrx/src/libhrx

CXX_FLAGS := -std=c++20 -fPIC -Wall -DUSEAVX2=1
CXX_FLAGS += -mavx2 -mfma -march=native -ffast-math
CXX_FLAGS += -fmax-errors=1
CXX_FLAGS += -I../../include
CXX_FLAGS += $(HRX_INC)
CXX_FLAGS += -MMD -MP
CXX_FLAGS += -DDEV_BUILD
CXX_FLAGS += -fopenmp
CXX_FLAGS += -DCMAKE_INSTALL_PREFIX="\"/opt/fastflowlm\""
CXX_FLAGS += -DCMAKE_XCLBIN_PREFIX="\"/opt/fastflowlm/share/flm/xclbins\""
#NOTE: TODO: FIXME: Either deprecate makefile, or keep the parameter sync with ../CMAKELists.txt, otherwise it is error-prone
CXX_FLAGS += -D__FLM_VERSION__="\"0.9.34\""
CXX_FLAGS += -D__NPU_VERSION__="\"32.0.203.304\""
CXX_FLAGS += -DDISABLE_ABI_CHECK=1

LDFLAGS += $(HRX_RPATH)
LDFLAGS += -lboost_program_options -lboost_filesystem
LDFLAGS += -L$(LIB_DIR)
LDFLAGS += -L../../build/tokenizers-cpp
LDFLAGS += -L../../build/tokenizers-cpp/sentencepiece/src
LDFLAGS += -ltokenizers_cpp -ltokenizers_c -lsentencepiece
LDFLAGS += $(HRX_LIBS)
DEPENDENCY_LDFLAGS += -lmha -ldequant -lgemm -llm_head -lq4_npu_eXpress


SOURCES += ../../common/utils.cpp

else

# WSL build environment
# Use CMake to invoke the Visual Studio
PWSH := powershell.exe

endif 

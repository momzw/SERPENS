# Makefile for serpens_hotloop

PYTHON ?= python3
CC ?= gcc

BASE_CFLAGS ?= -O3 -Wall -Wextra
CFLAGS ?= $(BASE_CFLAGS)
CPPFLAGS ?=
LDFLAGS ?=
LDLIBS ?= -lm

UNAME_S := $(shell uname -s 2>/dev/null || echo Unknown)

ifeq ($(OS),Windows_NT)
    PLATFORM := Windows
else ifneq (,$(findstring MINGW,$(UNAME_S)))
    PLATFORM := Windows
else ifneq (,$(findstring MSYS,$(UNAME_S)))
    PLATFORM := Windows
else ifeq ($(UNAME_S),Darwin)
    PLATFORM := Darwin
else ifeq ($(UNAME_S),Linux)
    PLATFORM := Linux
else
    PLATFORM := Unknown
endif

TARGET_BASENAME := serpens_hotloop
ifeq ($(PLATFORM),Windows)
    TARGET_EXT := .dll
else
    TARGET_EXT := .so
endif
TARGET := $(TARGET_BASENAME)$(TARGET_EXT)

REBOUND_DIR := $(shell $(PYTHON) -c "import os,rebound; print(os.path.dirname(rebound.__file__))" 2>/dev/null)
REBOUNDX_DIR := $(shell $(PYTHON) -c "import os,reboundx; print(os.path.dirname(reboundx.__file__))" 2>/dev/null)

ifeq ($(strip $(REBOUND_DIR)),)
    $(error Could not locate the Python package 'rebound'. Install it or set PYTHON=/path/to/python)
endif
ifeq ($(strip $(REBOUNDX_DIR)),)
    $(error Could not locate the Python package 'reboundx'. Install it or set PYTHON=/path/to/python)
endif

CPPFLAGS += -I"$(REBOUND_DIR)" -I"$(REBOUNDX_DIR)" -I"$(REBOUNDX_DIR)/reboundx"

ifeq ($(PLATFORM),Darwin)
    CFLAGS += -fPIC -D_GNU_SOURCE
    LDFLAGS += -shared -undefined dynamic_lookup
    LDLIBS += -lpthread
else ifeq ($(PLATFORM),Linux)
    CFLAGS += -fPIC -D_GNU_SOURCE
    LDFLAGS += -shared -Wl,--allow-shlib-undefined
    LDLIBS += -lpthread
else ifeq ($(PLATFORM),Windows)
    LDFLAGS += -shared
else
    $(error Unsupported platform '$(UNAME_S)'. Supported: Linux, macOS, Windows (MinGW/MSYS))
endif

all: $(TARGET)
build: all

$(TARGET): serpens_hotloop.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

clean:
	rm -f serpens_hotloop.so serpens_hotloop.dll

.PHONY: all build clean
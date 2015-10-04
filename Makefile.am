warning_flags = -Wall -Wextra -Wno-unused-parameter -Wshadow
common_cflags = -O2 $(MFLAGS) $(warning_flags)
AM_CXXFLAGS = -std=c++11 $(common_cflags)

AM_CPPFLAGS = $(VapourSynth_CFLAGS)

lib_LTLIBRARIES = libawarpsharp2.la

libawarpsharp2_la_SOURCES = src/awarpsharp2.cpp

if AWARPSHARP2_X86
libawarpsharp2_la_SOURCES += src/sse2.cpp
endif

libawarpsharp2_la_LDFLAGS = -no-undefined -avoid-version $(PLUGINLDFLAGS)

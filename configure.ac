AC_INIT([AWarpSharp2], [1], [https://github.com/dubhater/vapoursynth-awarpsharp2/issues], [AWarpSharp2], [https://github.com/dubhater/vapoursynth-awarpsharp2/])

: ${CXXFLAGS=""}

AM_INIT_AUTOMAKE([foreign no-dist-gzip dist-xz subdir-objects no-define])
AM_SILENT_RULES([yes])

LT_INIT([disable-static win32-dll])

AC_PROG_CXX

AC_CANONICAL_HOST


X86="false"
PPC="false"
ARM="false"

AS_CASE(
  [$host_cpu],
  [i?86],     [BITS="32" X86="true"],
  [x86_64],   [BITS="64" X86="true"],
  [powerpc*], [PPC="true"],
  [arm*],     [ARM="true"]
)

AS_CASE(
   [$host_os],
   [cygwin*|mingw*],
   [
      AS_IF(
         [test "x$BITS" = "x32"],
         [
          AC_SUBST([PLUGINLDFLAGS], ["-Wl,--kill-at"])
          AC_SUBST([STACKREALIGN], ["-mstackrealign"])
         ]
      )
   ]
)

AS_IF(
      [test "x$X86" = "xtrue"],
      [
       AC_DEFINE([AWARPSHARP2_X86])

       AC_SUBST([MFLAGS], ["-mfpmath=sse -msse2"])
      ]
)

AS_IF(
      [test "x$PPC" = "xtrue"],
      [AC_DEFINE([AWARPSHARP2_PPC])]
)

AS_IF(
      [test "x$ARM" = "xtrue"],
      [AC_DEFINE([AWARPSHARP2_ARM])]
)


AM_CONDITIONAL([AWARPSHARP2_X86], [test "x$X86" = "xtrue"])
AM_CONDITIONAL([AWARPSHARP2_ARM], [test "x$ARM" = "xtrue"])
AM_CONDITIONAL([AWARPSHARP2_PPC], [test "x$PPC" = "xtrue"])


PKG_CHECK_MODULES([VapourSynth], [vapoursynth])

AC_CONFIG_FILES([Makefile])
AC_OUTPUT

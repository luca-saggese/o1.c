/* jconfig.h — hand-made configuration for the vendored libjpeg-turbo 2.1.5
 * headers (jpeglib.h/jmorecfg.h/jerror.h). Matches the system
 * /lib/aarch64-linux-gnu/libjpeg.so.8 (libjpeg-turbo 2.1.5) so that the
 * struct layouts used here are ABI-compatible with the dlopen'd library.
 */
#ifndef JCONFIG_H
#define JCONFIG_H

/* Version ID for the JPEG library. */
#define JPEG_LIB_VERSION 80

/* libjpeg-turbo version */
#define LIBJPEG_TURBO_VERSION 2.1.5

/* libjpeg-turbo version in integer form */
#define LIBJPEG_TURBO_VERSION_NUMBER 2001005

/* Support arithmetic encoding */
#define C_ARITH_CODING_SUPPORTED 1

/* Support arithmetic decoding */
#define D_ARITH_CODING_SUPPORTED 1

/* Support in-memory source/destination managers */
#define MEM_SRCDST_SUPPORTED 1

/* Use accelerated SIMD routines. */
#define WITH_SIMD 1

/* 8-bit sample values (the usual setting) */
#define BITS_IN_JSAMPLE 8

/* Define if your (broken) compiler shifts signed values as if they were
   unsigned. */
/* #undef RIGHT_SHIFT_IS_UNSIGNED */

#endif /* JCONFIG_H */
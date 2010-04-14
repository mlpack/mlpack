/**
 * @file basic_types.h
 *
 * Basic definitions for FASTLIB.
 * These only exist for compatibility.
 * We should migrate to using stdint.h types where necessary.
 * After all, this just typedefs stdint.h types to the ones FASTLIB was using.
 */

#ifndef FASTLIB_BASE_TYPE_H
#define FASTLIB_BASE_TYPE_H

#include <stdint.h>

typedef uint8_t uint8;
typedef int8_t int8;

typedef uint16_t uint16;
typedef int16_t int16;

typedef uint32_t uint32;
typedef int32_t int32;

typedef uint64_t uint64;
typedef int64_t int64;

// printf macros
#if __WORDSIZE == 64
  #define L64 "l"
#else
  #define L64 "ll"
#endif

// float32 and float64 were never once referenced in the codebase, so we will
// not bother with those (since stdint.h does not include that support)

#endif

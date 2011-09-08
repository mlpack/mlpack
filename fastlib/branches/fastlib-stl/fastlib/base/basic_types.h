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

/*
typedef uint8_t_t_t uint8_t_t;
typedef int8_t_t int8_t;

typedef uint16_t_t_t uint16_t_t;
typedef int16_t_t int16_t;

typedef uint32_t_t_t uint32_t_t;
typedef int32_t_t int32_t;

typedef uint64_t_t_t uint64_t_t;
typedef int64_t_t int64_t;
*/
// printf macros
#if __WORDSIZE == 64
  #define L64 "l"
#else
  #define L64 "ll"
#endif

// float32 and float64 were never once referenced in the codebase, so we will
// not bother with those (since stdint.h does not include that support)

#endif

// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file scale.h
 *
 * Configuration for dealing with problem scale.
 *
 * This contains hints as to what sizes the problems might be and what
 * integer types to use; for instance, to use 64-bit integers on 64-bit
 * machines if scale is defined as FL_SCALE_LARGE.
 *
 * FL_SCALE_NORMAL - Problems are as large as conveniently fits into the
 *   machine's natural integer sizes.  On all major systems in 2007 this
 *   is 32 bits.
 *
 * FL_SCALE_LARGE - Support problem sizes as large as the architecture can
 *   support.  Unless you have at least 2 billion data points, this will
 *   probably waste space, reducing cache efficiency.
 *
 * FL_SCALE_TOOLARGE - Support problem sizes that are too large to fit in
 *   RAM, i.e. more than 2 billion data points.  This means 64-bit everything.
 */

#ifndef FL_SCALE_H
#define FL_SCALE_H

#include "base/basic_types.h"

#if !defined(FL_SCALE_NORMAL) && !defined(FL_SCALE_LARGE) && !defined(FL_SCALE_TOOLARGE)
/** Normal problem scale size - up to 16 gigabytes of data. */
#define FL_SCALE_NORMAL /* assume normal unless otherwise noted */
#endif

#ifdef FL_SCALE_NORMAL
typedef int index_t_impl; /* normal sized datasets - usually 32-bit */
#define LI_IMPL ""
#endif

#ifdef FL_SCALE_LARGE
typedef ssize_t index_t_impl; /* as large as can possibly fit on my machine */
#define LI_IMPL "l"
#endif

#ifdef FL_SCALE_TOOLARGE
typedef int64 index_t_impl; /* too large to fit in RAM? */
#define LI_IMPL L64
#endif

/**
 * Index type used throughout FASTlib for array sizes and vector dimensions.
 *
 * This value will be 32-bit most of the time to save space, but will be
 * defined to 64-bit in cases where you expect to use very large datasets.
 */
typedef index_t_impl index_t;

/**
 * Length modifier for emitting index_t with printf.
 *
 * Example:
 * @code
 *   index_t i = 42;
 *   printf("%"LI"d\n", i);
 * @endcode
 */
#define LI LI_IMPL

#endif /* FL_SCALE_H */

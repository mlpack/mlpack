/* MLPACK 0.2
 *
 * Copyright (c) 2008, 2009 Alexander Gray,
 *                          Garry Boyer,
 *                          Ryan Riegel,
 *                          Nikolaos Vasiloglou,
 *                          Dongryeol Lee,
 *                          Chip Mappus, 
 *                          Nishant Mehta,
 *                          Hua Ouyang,
 *                          Parikshit Ram,
 *                          Long Tran,
 *                          Wee Chin Wong
 *
 * Copyright (c) 2008, 2009 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
/**
 * @file fortran.h
 *
 * Basic types for FORTRAN compatability.
 */

#ifndef BASE_FORTRAN_H
#define BASE_FORTRAN_H

/* These typedefs should work for all machines I'm aware of. */

/** FORTRAN integer type. */
typedef int f77_integer;
/** FORTRAN Boolean type with values F77_TRUE and F77_FALSE. */
typedef unsigned int f77_logical;
/** FORTRAN single-precision type (e.g. float). */
typedef float f77_real;
/** FORTRAN double-precision type (e.g. double). */
typedef double f77_double;

/**
 * FORTRAN void return value.
 *
 * FORTRAN subroutines will still be prototyped to return an int, but this
 * integer must be ignored.
 */
typedef int f77_ret_void;
/** FORTRAN integer return value. */
typedef f77_integer f77_ret_integer;
/** FORTRAN Boolean return value. */
typedef f77_logical f77_ret_logical;
/**
 * FORTRAN single-precision return value.
 *
 * Note that FORTRAN seems to return doubles even for single-precision
 * functions.
 */
typedef f77_double f77_ret_real;
/** FORTRAN double-precision return value. */
typedef f77_double f77_ret_double;

/** FORTRAN single-precision complex number. */
typedef struct {
  f77_real re;
  f77_real im;
} f77_complex;
/** FORTRAN double-precision complex number. */
typedef struct {
  f77_double re;
  f77_double im;
} f77_doublecomplex;

/** Length of a FORTRAN string. */
typedef long f77_str_len;

/** False value for f77_logical type. */
#define F77_FALSE ((f77_logical)0)
/** True value for f77_logical type. */
#define F77_TRUE (~F77_FALSE)

/**
 * Does name-mangling for FORTRAN functions.
 *
 * Example:
 * @code
 *   F77_FUNC(fname)(a, b, c, d);
 * @endcode
 * translates to:
 * @code
 *   fname_(a, b, c, d);
 * @endcode
 */
#define F77_FUNC(fname) fname ## _

#ifdef __cplusplus
#define F77_UNKNOWN_ARGS ...
#else
#define F77_UNKNOWN_ARGS
#endif

/** FORTRAN function-pointer for integers. */
typedef f77_ret_integer (*f77_integer_func)( F77_UNKNOWN_ARGS );
/** FORTRAN function-pointer for Booleans. */
typedef f77_ret_logical (*f77_logical_func)( F77_UNKNOWN_ARGS );
/** FORTRAN function-pointer for single-precision. */
typedef f77_ret_real (*f77_real_func)( F77_UNKNOWN_ARGS );
/** FORTRAN function-pointer for double-precision. */
typedef f77_ret_double (*f77_double_func)( F77_UNKNOWN_ARGS );

#endif /* BASE_FORTRAN_H */

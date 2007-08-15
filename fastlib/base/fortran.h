/**
 * @file fortran.h
 *
 * Basic types for FORTRAN compatability.
 */

#ifndef BASE_FORTRAN_H
#define BASE_FORTRAN_H

// These typedefs should work for all machines I'm aware of.

/**
 * FORTRAN integer type.
 */
typedef int f77_integer;
/**
 * FORTRAN Boolean type for use with - use F77_TRUE and F77_FALSE.
 */
typedef unsigned int f77_logical;
/**
 * FORTRAN single-precision type (should be equivalent to float).
 */
typedef float f77_real;
/**
 * FORTRAN double-precision type (should be equivalent to double).
 */
typedef double f77_double;

/**
 * FORTRAN void return value.
 *
 * FORTRAN subroutines will still be prototyped to return an int, but this
 * integer must be ignored.
 */
typedef int f77_ret_void;

/**
 * FORTRAN integer return value.
 */
typedef f77_integer f77_ret_integer;
/**
 * FORTRAN Boolean return value.
 */
typedef f77_logical f77_ret_logical;
/**
 * FORTRAN single-precision return value.
 *
 * Note that FORTRAN seems to return doubles even for single-precision
 * functions.
 */
typedef double f77_ret_real;
/**
 * FORTRAN double-precision return value.
 */
typedef double f77_ret_double;

/**
 * Length of a FORTRAN string (when FORTRAN needs string lengths).
 */
typedef long f77_str_len;

#define F77_FALSE ((F77_LOGICAL)0)
#define F77_TRUE (~F77_FALSE)

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

/**
 * Does name-mangling for FORTRAN functions.
 *
 * Example:
 *
 * @code
 * F77_FUNC(somefn)(a, b, c, d);
 * @endcode
 *
 * translates to:
 *
 * @code
 * somefn_(a, b, c, d);
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

#endif

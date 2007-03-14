// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file compiler.h
 *
 * Defines compiler-specific optimizations.
 *
 * You will find the branch prediction macros likely and unlikely can make
 * certain parts of your code much faster; these profoundly speed up our
 * debugging mode.
 */

#ifndef BASE_COMPILER_H
#define BASE_COMPILER_H

/*
 * We delegate each macro to be implemented elsewhere.
 *
 * This allows us to document just one version of each macro (the public
 * one), and have multiple conditional implementations that aren't
 * documented.
 */

#define BASE_COMPILER_H__WANT_COMPILER_IMPL
#include "compiler_impl.h"
#undef BASE_COMPILER_H__WANT_COMPILER_IMPL

/**
 * Tells the compiler that C functions follow; use this at the beginning
 * of any list of C functions.
 *
 * This is necessary in C header files for C++ programs to link correctly
 * to the functions.
 */
#define EXTERN_C_START EXTERN_C_START__impl
/**
 * Tells the compiler that you are done listing C functions.
 */
#define EXTERN_C_END EXTERN_C_END__impl

/**
 * Tells the compiler to expect an expression to have this particular value
 * with high probability.
 *
 * This is useful for instance to predict the likely outcome of a branch,
 * but in that case you might just use like.y
 *
 * @param expr the expression being evaluated
 * @param value what you expect the expression to evaluate to
 * @see likely
 */
#define expect(expr, value) expect__impl(expr, value)

/**
 * Tells the compiler that a Boolean expression will most likely be true.
 *
 * @param x the expression that is usually be true
 */
#define likely(x) likely__impl(x)

/**
 * Tells the compiler that a Boolean expression will most likely be false.
 *
 * @param x the expression that is usually be false
 */
#define unlikely(x) unlikely__impl(x)

/**
 * Tells the compiler that a function never returns, such as exit or abort.
 *
 * Use this before your function, such as:
 *
 *   COMPILER_NORETURN void my_abort();
 */
#define COMPILER_NORETURN COMPILER_NORETURN__IMPL

/**
 * Tells the compiler to do parameter checking in a printf-like call.
 *
 * COMPILER_PRINTF(2, 4) void foo(int i, const char *format, int y, ...);
 *
 * In C++ member funtions, keep note that argument 1 is reserved for the
 * "this" parameter, so the first argument is argument 2.
 *
 * @param format_arg the argument number (starting with 1) of the format
 *        argument (in C++ member functions this starts with 2, see above)
 * @param dotdotdot_arg the number of the first variadic (...) parameter
 */
#define COMPILER_PRINTF(format_arg, dotdotdot_arg) \
    COMPILER_PRINTF__IMPL(format_arg, dotdotdot_arg)

/**
 * Tells the compiler that the function's result depends only on its
 * arguments, and there are zero visible side effects.
 *
 * This is useful for pretty much any pure-numerical function.  This allows
 * the compiler to optimize away repeated calls to a function, or to
 * rearrange the function calls freely.
 *
 * WARNING! Due to compiler bugs, *never* use this on functions that take in
 * pointers.
 *
 * COMPILER_FUNCTIONAL double my_hypot(double ax, double ay);
 */
#define COMPILER_FUNCTIONAL COMPILER_FUNCTIONAL__IMPL


/**
 * Computes the stride, or alignment, of a type.
 *
 * You must specify a type into strideof, not an actual variable.
 */
#define strideof(T) strideof__impl(T)

/**
 * Aligns a size_t to the stride of a particular type by roudning up.
 *
 * @param number the size_t to align
 * @param T the type
 * @return the stride
 */
#define stride_align(number, T) \
    (((size_t)(number) + strideof(T) - 1) / strideof(T) * strideof(T))

/**
 * Does the equivalent of static_cast, but works in C, while avoiding
 * warnings in C++.
 *
 * This is useful if you compile with -Wold-style-cast.
 */
#define STATIC_CAST(type, val) \
    COMPILER_CAST__impl(static_cast, type, val)

/**
 * Does the equivalent of reinterpret_cast, but works in C, while avoiding
 * warnings in C++.
 *
 * This is useful if you compile with -Wold-style-cast.
 */
#define REINTERPRET_CAST(type, val) \
    COMPILER_CAST__impl(reinterpret_cast, type, val)

#ifndef offsetof
#define offsetof(structure, field) ((size_t)((&(structure const *)0)->field))
#endif

#endif

// Copyright 2007 Georgia Institute of Technology. All rights reserved.
/**
 * @file debug.h
 *
 * Core antebugging support for FASTlib.
 *
 * We encourage you to leave debugging checks scattered throughout
 * your code.  Compiling with "--mode=fast" will eliminate all of
 * these checks, and the branch-prediction optimizations they use
 * minimze their impact on speed even in other modes.  Empirically,
 * debug checks only result in a 10-20% drop in performance, so we
 * recommend always compiling with them in unless performing speed
 * tests.
 *
 * Use VERBOSE_GOT_HERE and assertions often and initialize members of
 * reusable classes with BIG_BAD_NUMBER and DEBUG_POISON_PTR.
 */

#ifndef BASE_DEBUG_H
#define BASE_DEBUG_H

#include "common.h"

/** Performs an expression only in debug mode. */
#ifdef DEBUG
#define DEBUG_ONLY(x) (x)
#else
#define DEBUG_ONLY(x) NOP
#endif

/** Performs an expression only in verbose mode. */
#ifdef VERBOSE
#define VERBOSE_ONLY(x) (x)
#else
#define VERBOSE_ONLY(x) NOP
#endif

/** Performs an expression only in profile mode. */
#ifdef PROFILE
#define PROFILE_ONLY(x) (x)
#else
#define PROFILE_ONLY(x) NOP
#endif

/** Verbosity for VERBOSE_MSG and VERBOSE_GOT_HERE. */
extern double verbosity_level;
/** Whether to process VERBOSE_GOT_HERE. */
extern int print_got_heres;
/** Whether to process DEBUG_WARNING_MSG_IF and DEBUG_WARNING_IF. */
extern int print_warnings;

/**
 * Prints a message only in verbose mode at a given level of
 * verbosity.
 *
 * Calls to this macro may double as comments for your code and
 * produce no overhead outside of verbose mode.  Additionally,
 * messages will not be printed if verbosity_level is less than the
 * specified minimum.
 *
 * Example:
 *
 * @code
 * if (object.type == RABBIT) {
 *   VERBOSE_MSG(3.0, "Processing %s as a rabbit", object.name);
 *   process_rabbit(object);
 * }
 * @endcode
 *
 * @param min_verbosity level of verbosity required to emit mesage
 * @param msg_params format string and variables, as in printf
 */
#define VERBOSE_MSG(min_verbosity, msg_params...) \
    VERBOSE_ONLY( \
        unlikely(verbosity_level >= (min_verbosity)) \
            ? NOTIFY(msg_params) : NOP)

/**
 * Prints a default message to indicate having reached a line of code.
 *
 * These messages may be run-time disabled by unsetting global
 * print_got_heres.
 *
 * @param min_verbosity level of verbosity required to emit mesage
 */
#define VERBOSE_GOT_HERE(min_verbosity) \
    VERBOSE_ONLY( \
        unlikely(print_got_heres && verbosity_level >= (min_verbosity)) \
            ? NOTIFY("Got to line %d of %s", __LINE__, __FUNCTION__) : NOP)

/**
 * Conditionally emits a warning message, which may abort or pause
 * process.
 *
 * The condition is only tested when debug mode is active.  It is not
 * recommended for conditions to have side-effects.
 *
 * Warnings may be run-time disabled by unsetting global
 * print_warnings.
 *
 * @param cond the condition when the warning should be raised
 * @param msg_params format string and variables, as in printf
 */
#define DEBUG_WARNING_MSG_IF(cond, msg_params...) \
    DEBUG_ONLY(unlikely((cond) && print_warnings) \
        ? NONFATAL(msg_params) : NOP)

/**
 * Conditionally emits a default warning message, which may abort of
 * pause process.
 *
 * The condition is only tested when debug mode is active.  It is not
 * recommended for conditions to have side-effects.
 *
 * Warnings may be run-time disabled by unsetting global
 * print_warnings.
 *
 * @param cond the condition when the warning should be raised
 */
#define DEBUG_WARNING_IF(cond) \
    DEBUG_WARNING_MSG_IF(cond, "warning: " #cond)

/**
 * Conditionally emits an error message, aborting process.
 *
 * The condition is only tested when debug mode is active.  It is not
 * recommended for conditions to have side-effects.
 *
 * @param cond the condition when the error should be raised
 * @param msg_params format string and variables, as in printf
 */
#define DEBUG_ERROR_MSG_IF(cond, msg_params...) \
    DEBUG_ONLY(unlikely(cond) \
        ? FATAL(msg_params) : NOP)

/**
 * Conditionally emits a default error message, aborting process.
 *
 * The condition is only tested when debug mode is active.  It is not
 * recommended for conditions to have side-effects.
 *
 * @param cond the condition when the error should be raised
 */
#define DEBUG_ERROR_IF(cond) \
    DEBUG_ERROR_MSG_IF(cond, "error: " #cond)

/**
 * Aborts process if some condition fails, printing a given message.
 *
 * The condition is only tested when debug mode is active.  It is not
 * recommended for conditions to have side-effects.
 *
 * @param cond the condition that must be true to proceed
 * @param msg_params format string and variables, as in printf
 */
#define DEBUG_ASSERT_MSG(cond, msg_params...) \
    DEBUG_ONLY(likely(cond) \
        ? NOP : FATAL(msg_params))

/**
 * Aborts process if a condition fails, printing a standard message.
 *
 * The condition is only tested when debug mode is active.  It is not
 * recommended for conditions to have side-effects.
 *
 * @param cond the condition that must be true to proceed
 */
#define DEBUG_ASSERT(cond) \
    DEBUG_ASSERT_MSG(cond, "assertion failure: %s", #cond)

/**
 * An easy-to-spot invalid number, for use with debugging tools.
 *
 * This value is large enough that array allocation should fail and
 * iterating upto the value should segfault.  This bit pattern also is
 * 0x7FF388AA, or NaN as a float or the high bits of a double.
 */
#define BIG_BAD_NUMBER 2146666666

/** 
 * An obviously invalid pointer, for use with debugging tools.
 *
 * Use DEBUG_POISON_PTR to initialize a pointer to BIG_BAD_POINTER in
 * either C or C++ without having to specify the type.
 *
 * @param type the type of the pointer to be poisoned
 */
#define BIG_BAD_POINTER(type) (REINTERPRET_CAST(type *, 0xDEADBEEF))

/** Implementation for DEBUG_POISON_PTR. */
#ifdef __cplusplus
template<typename T>
const T *poison_ptr(T *&x) {
  return x = BIG_BAD_POINTER(T);
}
#else
#define poison_ptr(x) ((x) = BIG_BAD_POINTER(void))
#endif

/**
 * Sets a pointer to a BIG_BAD_POINTER in debug mode.
 *
 * May cause slow-down in debug mode because within-function pointers
 * may not be converted to registers by the compiler.  This is not an
 * issue for pointers to the heap or stored in classes or structs, as
 * these could not have been registers anyway.
 *
 * @param x the pointer to be poisoned
 */
#define DEBUG_POISON_PTR(x) DEBUG_ONLY(poison_ptr(x))

/**
 * Asserts that an index is positive and less than its upper bound.
 *
 * This macro converts inputs to uint64 to simultaneously test the
 * upper bound and positivity; negative numbers become large and
 * positive.  Accordingly, this macro may fail if the bound is a
 * gigantic unsigned value (too big to be signed) and x < -1.
 *
 * Expressions for x and bound are run a second time when reporting an
 * error (and not at all if not in debug mode) and thus should not
 * have side-effects or be computationally intensive.
 *
 * @param x the index value to test
 * @param bound the upper bound for x; 0 is the implicit lower bound
 */
#define DEBUG_BOUNDS(x, bound) \
    DEBUG_ASSERT_MSG(STATIC_CAST(uint64, x) < STATIC_CAST(uint64, bound), \
        "DEBUG_BOUNDS failed: %s = %"L64"d not in [0, %s = %"L64"d)\n", \
        #x, STATIC_CAST(int64, x), #bound, STATIC_CAST(int64, bound))

/**
 * Asserts than an index is positive and less than or equal to its
 * upper bound.
 *
 * This macro is nearly the same as DEBUG_BOUNDS, but permits the
 * index and upper bound to be equal.
 *
 * @param x the index value to test
 * @param bound the upper bound for x; 0 is the implicit lower bound
 */
#define DEBUG_BOUNDS_INCLUSIVE(x, bound) \
    DEBUG_ASSERT_MSG(STATIC_CAST(uint64, x) <= STATIC_CAST(uint64, bound), \
        "DEBUG_BOUNDS failed: %s = %"L64"d not in [0, %s = %"L64"d]\n", \
        #x, STATIC_CAST(int64, x), #bound, STATIC_CAST(int64, bound))

/**
 * Asserts that two integers are the same.
 *
 * Expressions for x and y are run a second time when reporting an
 * error (and not at all if not in debug mode) and thus should not
 * have side-effects or be computationally intensive.
 *
 * @param x left-hand side of the equality test
 * @param y right-hand side of the equality test
 */
#define DEBUG_SAME_INT(x, y) \
    DEBUG_ASSERT_MSG((x) == (y), \
        "DEBUG_SAME_INT failed: %s = %"L64"d not equal to %s = %"L64"d\n", \
        #x, STATIC_CAST(int64, x), #y, STATIC_CAST(int64, y))

/**
 * Asserts that two integers are the same.
 *
 * This is currently the same as DEBUG_SAME_INT, but is specifically
 * intended for use comparing array lengths.
 *
 * Expressions for x and y are run a second time when reporting an
 * error (and not at all if not in debug mode) and thus should not
 * have side-effects or be computationally intensive.
 *
 * @param x left-hand side of the equality test
 * @param y right-hand side of the equality test
 */
#define DEBUG_SAME_SIZE(x, y) \
    DEBUG_ASSERT_MSG((x) == (y), \
        "DEBUG_SAME_SIZE failed: %s = %"L64"d not equal to %s = %"L64"d\n", \
        #x, STATIC_CAST(int64, x), #y, STATIC_CAST(int64, y))

/**
 * Asserts that two floating-point numbers are the same.
 *
 * Expressions for x and y are run a second time when reporting an
 * error (and not at all if not in debug mode) and thus should not
 * have side-effects or be computationally intensive.
 *
 * @param x left-hand side of the equality test
 * @param y right-hand side of the equality test
 */
#define DEBUG_SAME_DOUBLE(x, y) \
    DEBUG_ASSERT_MSG((x) == (y), \
        "DEBUG_SAME_DOUBLE failed: %s = %g not equal to %s = %g\n", \
        #x, STATIC_CAST(double, x), #y, STATIC_CAST(double, y))

/**
 * Asserts that two floating-point numbers are within epsilon of each
 * other.
 *
 * Expressions for x and y are run a second time when reporting an
 * error (and not at all if not in debug mode) and thus should not
 * have side-effects or be computationally intensive.
 *
 * @param x left-hand side of the equality test
 * @param y right-hand side of the equality test
 */
#define DEBUG_APPROX_DOUBLE(x, y, eps) \
    DEBUG_ASSERT_MSG(fabs(STATIC_CAST(double, (x) - (y))) > eps, \
        "DEBUG_APPROX_DOUBLE failed: %s = %g not within %g of %s = %g\n", \
        #x, STATIC_CAST(double, x), STATIC_CAST(double, eps), \
        #y, STATIC_CAST(double, y))

#endif /* BASE_DEBUG_H */

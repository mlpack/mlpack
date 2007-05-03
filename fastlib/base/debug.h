// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file debug.h
 *
 * Core antebugging support for FASTlib.
 *
 * We encourage you to scatter all your code with debug code to help
 * you find problems, and leave these checks.  Use DEBUG_GOT_HERE and
 * DEBUG_ERR_IF everywhere.  If you write reusable classes, in your
 * default constructors, initialize things to BIG_BAD_NUMBER and use
 * DEBUG_POISON_PTR to poison pointers with 0xDEADBEEF.
 *
 * One of FASTlib's main design points is that it takes advantage of
 * the speed of C++ while providing minimal-overhead support for
 * runtime debugging.  We usually don't see more than about a 10% or
 * 20% drop in performance due to branch prediction directives
 * 'likely' and 'unlikely'.  In fact, we encourage you to always
 * compile with debugging activated except when performing empirical
 * timing analysis.  We were careful in writing this to avoid
 * performance drops; turning off the branch prediction hint macros in
 * compiler.h may performance to drop by as much as 60%.
 */

#ifndef BASE_DEBUG_H
#define BASE_DEBUG_H

#include "compiler.h"
#include "common.h"

#include <stdio.h>
#include <assert.h>
#include <libgen.h>

/**
 * Perform an expression only in debug mode.
 */
#ifdef DEBUG
#define DEBUG_ONLY(x) (x)
#else
#define DEBUG_ONLY(x) NOP
#endif

/**
 * Perform an expression only in verbose mode.
 */
#ifdef VERBOSE
#define VERBOSE_ONLY(x) (x)
#else
#define VERBOSE_ONLY(x) NOP
#endif

/** Verbosity for statements printed with DEBUG_MSG and DEBUG_GOT_HERE. */
extern double debug_verbosity;

/** Whether to process calls to DEBUG_GOT_HERE. */
extern int print_got_heres;

/** Whether to process calls to DEBUG_WARN_MSG_IF and DEBUG_WARN_IF. */
extern int print_warnings;

/**
 * Print a message only in debug mode at a given level of verbosity.
 *
 * Calls to this macro may double as comments for your code and
 * produce no overhead when not compiled in verbose mode.
 * Additionally, messages will not be printed (or evaluted) if
 * debug_verbosity is less than the specified minimum.
 *
 * Example:
 *
 * @code
 * if (object.type == RABBIT) {
 *   DEBUG_MSG(3.0, "Processing %s as a rabbit", object.name);
 *   process_rabbit(object);
 * }
 * @endcode
 *
 * @param min_verbosity level of verbosity required to emit mesage
 * @param msg_params format string and variables, as in printf
 */
#define DEBUG_MSG(min_verbosity, msg_params...) \
    VERBOSE_ONLY( \
        unlikely(debug_verbosity >= (min_verbosity)) \
            ? NOTIFY(msg_params) : NOP)

/**
 * Print a default message to indicate having reached a line of code.
 *
 * These messages may be run-time disabled by unsetting global
 * print_got_heres.
 *
 * @param min_verbosity level of verbosity required to emit mesage
 */
#define DEBUG_GOT_HERE(min_verbosity) \
    VERBOSE_ONLY( \
        unlikely(print_got_heres && debug_verbosity >= (min_verbosity)) \
            ? NOTIFY("Got to line %d of %s", __LINE__, __func__) : NOP)

/**
 * Conditionally emit a warning message, which may abort or pause
 * process.
 *
 * The condition is only tested when debug mode is active.  It is not
 * recommended to run code with side-effects in the condition.
 *
 * Warnings may be run-time disabled by unsetting global
 * print_warnings.
 *
 * @param cond the condition upon which a warning should be raised
 * @param msg_params format string and variables, as in printf
 */
#define DEBUG_WARN_MSG_IF(cond, msg_params...) \
    DEBUG_ONLY(unlikely((cond) && print_warnings) \
        ? NONFATAL(msg_params) : NOP)

/**
 * Conditionally emit a default warning message.
 *
 * @param cond the condition upon which a warning should be raised
 */
#define DEBUG_WARN_IF(cond) \
    DEBUG_WARN_MSG_IF(cond, "warning: " #cond)

/**
 * Conditionally emit an error message, aborting process.
 *
 * The condition is only tested when debug mode is active.  It is not
 * recommended to run code with side-effects in the condition.
 *
 * @param cond the condition upon which an error should be raised
 * @param msg_params format string and variables, as in printf
 */
#define DEBUG_ERR_MSG_IF(cond, msg_params...) \
    DEBUG_ONLY(unlikely(cond) \
        ? FATAL(msg_params) : NOP)

/**
 * Conditionally emit a default error message.
 *
 * @param cond the condition upon which a error should be raised
 */
#define DEBUG_ERR_IF(cond) \
    DEBUG_ERR_MSG_IF(cond, "error: " #cond)

/**
 * Abort process if some condition fails, printing a given message.
 *
 * The message is formatted printf-style.
 *
 * @param cond the condition that must be true to proceed
 */
#define DEBUG_ASSERT_MSG(cond, msg_params...) \
    DEBUG_ONLY(likely(cond) \
        ? NOP : FATAL(msg_params))

/**
 * Aborts process if a condition fails.
 *
 * @param cond the condition that must be true to proceed
 */
#define DEBUG_ASSERT(cond) \
    DEBUG_ASSERT_MSG(cond, "assertion failure: " #cond)

/**
 * An easy-to-spot invalid number, for use with debugging tools.
 *
 * This value is large enough that array allocation should fail and
 * iterating upto the value should segfault.  This bit pattern also is
 * 0x7FF388AA, or NaN as a float or double.
 */
#define BIG_BAD_NUMBER 2146666666

/**
 * An obviously invalid pointer, for use with debugging tools.
 */
#define BIG_BAD_POINTER(type) (REINTERPRET_CAST(type*, 0xDEADBEEF))

/**
 * Sets a pointer to a BIG_BAD_POINTER in debug mode.
 *
 * In debug mode, this will make your code run slower if the pointer is
 * declared inside the current function, because it will prevent the
 * compiler from making it a register.  That is the price you pay for
 * debugging.  One day this will save your code, guaranteed.
 */
#ifdef __cplusplus
#define DEBUG_POISON_PTR(x) \
    DEBUG_ONLY(debug_poison_ptr__impl(x))

template<typename T>
void debug_poison_ptr__impl(T *&x) {
  x = BIG_BAD_POINTER(T);
}
#else
#define DEBUG_POISON_PTR(x) \
    DEBUG_ONLY(var = BIG_BAD_POINTER(void))
#endif

/**
 * Makes sure an integer is positive and less than some upper bound.
 *
 * This macro converts inputs to uint64 in order to simultaneously
 * test whether x is negative.  Expressions with side-effects should
 * not be supplied for x or bound because this code is run a second
 * time when reporting the error (and because this code is not run
 * unless in debug mode).
 *
 * @param x the value to be tested
 * @param bound a number that should be greater than x
 */
#define DEBUG_BOUNDS(x, bound) \
    DEBUG_ASSERT_MSG(STATIC_CAST(uint64, x) < STATIC_CAST(uint64, bound), \
        "%s == %"L64"d exceeds bound %s == %"L64"d\n", \
        #x, STATIC_CAST(uint64, x), #bound, STATIC_CAST(uint64, bound))

/**
 * Asserts that two integers are the same in debug mode.
 */
#define DEBUG_SAME_INT(a, b) \
    DEBUG_ASSERT_MSG(((a) == (b)), "[%s] %d != %d [%s]", #a, (int)(a), (int)(b), #b)

#endif

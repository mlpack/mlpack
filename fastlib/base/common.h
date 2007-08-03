// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file common.h
 *
 * Includes all the bare necessities, like types, debugging, and some
 * convenience items.
 *
 * @see compiler.h
 * @see debug.h
 * @see scale.h
 */

#ifndef BASE_COMMON_H
#define BASE_COMMON_H

#ifndef _REENTRANT
#define _REENTRANT
#endif

#ifdef __cplusplus
#include "cc.h"
#endif

#include "base/basic_types.h"
#include "compiler.h"
#include "debug.h"
#include "scale.h"

#include <float.h>
#include <math.h>

EXTERN_C_START

/** A suitable no-op for macro expressions. */
#define NOP ((void)0)

/**
 * Currently, equivalent to C's abort().
 *
 * Note that built-in abort() already flushes all streams.
 */
#define fl_abort() abort()

/**
 * Waits for the user to press return.
 *
 * This function will obliterate anything in the stdin buffer.
 */
void fl_pause(void);

/**
 * Prints out a blue-colored percentage status indicator.
 */
void percent_indicator(const char *what, uint64 numerator, uint64 denominator);

/**
 * Obtains a more concise filename from a full path.
 *
 * If the file is in the FASTlib directories, this will shorten the
 * path to contain only the portion beneath the c directory.
 * Otheerwise, it will give the file name and its containing
 * directory.
 */
const char *fl_filename(const char* name);

/**
 * Prints a message header for a specified location in code.
 */
void fl_msg_header(char type, const char *color,
    const char *file, const char *func, int line);

/** Whether to treat nonfatal warnings as fatal. */
extern int abort_on_nonfatal;

/** Whether to wait for user input after nonfatal warnings. */
extern int pause_on_nonfatal;

/** Whether to print call locations for notifications. */
extern int print_notify_headers;

/** Implementation for FATAL - use FATAL instead. */
COMPILER_NORETURN
COMPILER_PRINTF(4, 5)
void fatal(const char *file, const char *func, int line,
	   const char* format, ...);

/** Implementation for NONFATAL - use NONFATAL instead. */
COMPILER_PRINTF(4, 5)
void nonfatal(const char *file, const char *func, int line,
	      const char* format, ...);

/** Implementation for NOTIFY - use NOTIFY instead. */
COMPILER_PRINTF(4, 5)
void notify(const char *file, const char *func, int line,
	    const char* format, ...);

/**
 * Aborts, printing call location and a message.
 *
 * Message is sent to stderr.  Arguments are equivalent to printf.
 */
#define FATAL(msg_params...) \
    (fatal(__FILE__, __func__, __LINE__, msg_params))

/**
 * (Possibly) aborts or pauses, printing call location and a message.
 *
 * Message is sent to stderr.  Arguments are equivalent to printf.
 */
#define NONFATAL(msg_params...) \
    (nonfatal(__FILE__, __func__, __LINE__, msg_params))

/**
 * Prints (possibly) call location and a message.
 *
 * Message is sent to stderr.  Arguments are equivalent to printf.
 */
#define NOTIFY(msg_params...) \
    (notify(__FILE__, __func__, __LINE__, msg_params))

/** The maximum number of tsprintf buffers available at any time. */
#define TSPRINTF_COUNT 16

/** The maximum length of a tsprintf string. */
#define TSPRINTF_LENGTH 512

/**
 * DEPRECATED - NOT SAFE.
 *
 * Formats a string and returns an ultra-convenient string which you
 * do not have to free, and can use for a limited time.  If you simply
 * pass this to another function you will be safe.
 *
 * This works by maintaining a pool of TSPRINTF_COUNT buffers which
 * are reused circularly.
 *
 * If the string is longer than TSPRINTF_LENGTH, it will be truncated.
 */
COMPILER_PRINTF(1, 2)
char *tsprintf(const char *format, ...);

/**
 * Standard return value for indicating success or failure.
 *
 * It is recommended to use these consistently, as they have a fixed meaning.
 * Unfortunately, many functions indicate failure with nonzero return values,
 * whereas some functions indicate success with nonzero return values.
 */
typedef enum {
  /**
   * Return value for indicating successful operation.
   */
  SUCCESS_PASS = 20,
  /**
   * Return value indicating an operation may have been successful, but
   * there are things that a careful programmer or user should be wary of.
   */
  SUCCESS_WARN = 15,
  /**
   * Return value indicating an operation failed.
   */
  SUCCESS_FAIL = 10
} success_t;


/**
 * Asserts that a particular operation passes, otherwise terminates
 * the program.
 *
 * This check will *always* occur, regardless of debug mode.  It is suitable
 * for handling functions that return error codes, where it is not worth
 * trying to recover.
 */
#define MUST_PASS(x) ((likely(x >= SUCCESS_PASS)) ? NOP \
         : FATAL("MUST_PASS failed: %s", #x))

/**
 * @deprecated
 *
 * The terminology of this is confusing, because this looks like it's
 * for debug-mode, but it is not.
 *
 * Instead use MUST_PASS which acts exactly the same.
 */
#define ASSERT_PASS(x) MUST_PASS(x)

/**
 * Turns return values from most C standard library functions into a
 * standard success_t value, under the assumption that negative is bad.
 *
 * If x is less than 0, failure is assumed; otherwise, success is assumed.
 */
#define SUCCESS_FROM_INT(x) (unlikely((x) < 0) ? SUCCESS_FAIL : SUCCESS_PASS)

/**
 * Returns true if something passed, false if it failed or incurred warnings.
 *
 * Branch-predictor-optimized for passing case.
 */
#define PASSED(x) (likely((x) >= SUCCESS_PASS))

/**
 * Returns true if something failed, false if it passed or incurred warnings.
 *
 * Branch-predictor-optimized for non-failing case.
 */
#define FAILED(x) (unlikely((x) <= SUCCESS_FAIL))

EXTERN_C_END

/** ANSI color sequence wrapper */
#define ANSI_SEQ(str) "\033["str"m"
/** Clears ANSI colors */
#define ANSI_CLEAR ANSI_SEQ("0")
/** Begin high-intensity */
#define ANSI_BOLD ANSI_SEQ("1")

/** Color code: High-intensity Black */
#define ANSI_HBLACK ANSI_SEQ("1;30")
/** Color code: High-intensity Red */
#define ANSI_HRED ANSI_SEQ("1;31")
/** Color code: High-intensity Green */
#define ANSI_HGREEN ANSI_SEQ("1;32")
/** Color code: High-intensity Yellow */
#define ANSI_HYELLOW ANSI_SEQ("1;33")
/** Color code: High-intensity Blue */
#define ANSI_HBLUE ANSI_SEQ("1;34")
/** Color code: High-intensity Magenta */
#define ANSI_HMAGENTA ANSI_SEQ("1;35")
/** Color code: High-intensity Cyan */
#define ANSI_HCYAN ANSI_SEQ("1;35")
/** Color code: High-intensity White */
#define ANSI_HWHITE ANSI_SEQ("1;36")

/** Color code: Black */
#define ANSI_BLACK ANSI_SEQ("30")
/** Color code: Red */
#define ANSI_RED ANSI_SEQ("31")
/** Color code: Green */
#define ANSI_GREEN ANSI_SEQ("32")
/** Color code: Yellow */
#define ANSI_YELLOW ANSI_SEQ("33")
/** Color code: Blue */
#define ANSI_BLUE ANSI_SEQ("34")
/** Color code: Magenta */
#define ANSI_MAGENTA ANSI_SEQ("35")
/** Color code: Cyan */
#define ANSI_CYAN ANSI_SEQ("35")
/** Color code: White */
#define ANSI_WHITE ANSI_SEQ("36")

#endif

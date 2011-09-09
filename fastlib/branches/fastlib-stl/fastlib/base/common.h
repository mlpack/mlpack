/**
 * @file common.h
 *
 * The bare necessities of FASTlib programming in C, including
 * standard types, formatted messages to stderr, and useful libraries
 * and compiler directives.
 *
 * This file should be included before all built-in libraries because
 * it includes the _REENTRANT definition needed for thread-safety.
 * Files common.h or fastlib.h include this file first and may serve as
 * surrogates.
 *
 * @see compiler.h
 */

#ifndef BASE_COMMON_H
#define BASE_COMMON_H

#ifndef _REENTRANT
#define _REENTRANT
#endif

#include "compiler.h"
#include "ansi_colors.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include <stdint.h>

#endif /* BASE_COMMON_H */

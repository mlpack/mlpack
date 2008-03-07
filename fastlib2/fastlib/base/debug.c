/**
 * @file debug.c
 *
 * Default variable settings for debug options.
 */

#include "fastlib/base/debug.h"

double verbosity_level = 0.0;
int print_got_heres = 1;
int print_warnings = 1;

const int32 BIG_BAD_BUF[] = {
  BIG_BAD_NUMBER,
  BIG_BAD_NUMBER,
  BIG_BAD_NUMBER,
  BIG_BAD_NUMBER,
  BIG_BAD_NUMBER,
  BIG_BAD_NUMBER,
  BIG_BAD_NUMBER,
  BIG_BAD_NUMBER,
  BIG_BAD_NUMBER,
  BIG_BAD_NUMBER,
  BIG_BAD_NUMBER,
  BIG_BAD_NUMBER,
  BIG_BAD_NUMBER,
  BIG_BAD_NUMBER,
  BIG_BAD_NUMBER,
  BIG_BAD_NUMBER
};

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
 * @file test.h
 *
 * Very basic unit-testing infrastructure.
 *
 * WARNING: This is likely to be deprecated at any time to be more
 * sophisticated! Use at your own risk.
 */

#ifndef BASE_TEST_H
#define BASE_TEST_H

#include "fastlib/base/common.h"

/**
 * Make a testing assertion.
 *
 * (This may eventually be more sophisticated)
 */
#define TEST_ASSERT(x) \
    (likely(x) ? NOP : FATAL("Assertion failed: %s", #x))

#define TEST_DOUBLE_EXACT(a, b) \
    if (unlikely((a) != (b))) \
    FATAL("%.10e (%s) != %.10e (%s)", (double)(a), #a, (double)(b), #b); else

#define TEST_DOUBLE_APPROX(a, b, absolute_eps) \
    if (unlikely(fabs((a) - (b)) > absolute_eps)) \
    FATAL("%.10e (%s) !~= %.10e (%s)", (double)(a), #a, (double)(b), #b); else

/**
 * Begin a test suite of a given name.
 *
 * After this, declare a lot of void functions that contain assertions.
 */
#define TEST_SUITE_BEGIN(suite_name) \
    namespace { /* begin the private namespace */

/** Prototype for test functions -- take in no arguments */
typedef void (*test__void_func)();

/**
 * End a test suite of a given name, and generate a main.
 */
#define TEST_SUITE_END(suite_name, functions ...) \
      int execute_tests(int which_test) { \
        test__void_func tests[] = { functions }; \
        int n_tests = sizeof(tests) / sizeof(tests[0]); \
        \
        if (which_test < 0) { \
          for (int i = 0; i < n_tests; i++) { \
              tests[i](); \
          } \
        } else { \
          if (which_test >= n_tests) { \
            fprintf(stderr, "NO MORE TESTS!\n"); \
            return 3; \
          } \
        } \
        fprintf(stderr, "ALL TESTS PASSED!\n"); \
        return 0; \
      } \
    }; /* end the private namespace */ \
    int main(int argc, char *argv[]) { \
      return execute_tests( \
         argc <= 1 ? -1 : atoi(argv[1])); \
    }

#endif

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

#include "base/common.h"

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
    namespace suite_name ## _test { /* begin the private namespace */

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
      return suite_name ## _test :: execute_tests( \
         argc <= 1 ? -1 : atoi(argv[1])); \
    }

#endif

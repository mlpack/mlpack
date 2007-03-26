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

/* Debug mode must be on for these tests to work -- so we make sure the
 * compiler will fail if DEBUG is not enabled.
 */
#ifdef DEBUG

/**
 * Make a testing assertion.
 *
 * (This may eventually be more sophisticated)
 */
#define TEST_ASSERT(x) \
    DEBUG_ASSERT(x)

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
    namespace suite_name ## _test {

/**
 * End a test suite of a given name, and generate a main.
 */
#define TEST_SUITE_END(suite_name, functions ...) \
      int execute_all_tests() { \
        test__void_func tests[] = { functions }; \
        for (size_t i = 0; i < sizeof(tests)/sizeof(tests[0]); i++) { \
            tests[i](); \
        } \
        fprintf(stderr, "ALL TESTS PASSED!\n"); \
        return 0; \
      } \
    }; \
    int main(void) { \
      return suite_name ## _test :: execute_all_tests(); \
    }

typedef void (*test__void_func)();

#endif

#endif

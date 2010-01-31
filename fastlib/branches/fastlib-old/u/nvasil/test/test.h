/*
 * =====================================================================================
 * 
 *       Filename:  test.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  04/27/2007 09:49:58 AM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
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
 * Create a test suite of a given name, 
 */
#define TEST_SUITE(functions ...) \
typedef   (*test__void_func)();\
int execute_all_tests() { \
        test__void_func tests[] = { functions }; \
        for (size_t i = 0; i < sizeof(tests)/sizeof(tests[0]); i++) { \
            Init();\
					  tests[i](); \
					  Destruct(); \
        } \
        fprintf(stderr, "ALL TESTS PASSED!\n"); \
        return 0; \
     } \
    

/**
 * Run the tests
 */
#define RUN_TESTS(test_class) \
  int main() { \
			test_class test; \
      return test.execute_all_tests(); \
    }


#define AFFILIATE_WITH_TESTER(class_name) \
	friend class class_name ## Test;

#endif

#endif
    

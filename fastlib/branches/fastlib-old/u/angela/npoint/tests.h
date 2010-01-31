/** 
 * @author: Angela Grigoroaia
 * @file: tests.h
 * 
 * @description: Simple unit tests for npoint.
 */

#ifndef TESTS_H
#define TEST_H

/**
 * Main function that decides which tests need to be run and what arguments are
 * required in order for the tests to be ok. The function has the folowing
 * pattern:
 * 		1. Do I run this test?
 * 		2. If I do 			=> Get all relevant parameters?
 * 										=> Run tests and output results.
 * 		3. If I don't		=> Move to the next test.
 */
void test_main(String test);

/** 
 * This can be used to test if we are properly generating all the possible 
 * permutations of size n. It counts all the permutations that we generate and
 * checks if the final value is n!. 
 */
success_t test_permutation_generator(int n);

/**
 * These can be used to test if the ntuple index generators are correct. For N
 * points we should generate S(N,n) = (N choose n) ntuples in the symmetric case
 * and A(N,n) = S(N,n)*n! for the asymmetric case. The functions simply count
 * the number of generated indexes and check the count agains the theoretical
 * values.
 */
success_t test_symmetric_index_generator(int n, index_t N);
success_t test_asymmetric_index_generator(int n, index_t N);

/**
 * This is a simple implementation of n choose k used primarily for testing the
 * index generators.
 */
double choose(int n, int k);


#endif

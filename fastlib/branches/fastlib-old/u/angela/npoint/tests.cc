/**
 * @author: Angela Grigoroaia
 * @file: tests.cc
 *
 * @description: Simple unit tests for npoint.
 */

#include "fastlib/fastlib.h"
#include "globals.h"
#include "datapack.h"
#include "metrics.h"
#include "matcher.h"
#include "naive.h"

#include "tests.h"

/*****************************************************************************/

void test_main(String test) {

	/* Index generators */
	if ( !test.CompareTo("indexes") ) {
		const int n = fx_param_int(NULL,"n",2);
		const int npoints = fx_param_int_req(NULL,"npoints");

		if ( !PASSED(test_symmetric_index_generator(n,npoints)) ) {
			fprintf(output,"\n\nSymmetric index generator failed for n = %d and	npoints = %d\n",n,npoints);
		}
		else {
			fprintf(output,"\n\nSymmetric index generator worked for n = %d and	npoints = %d\n",n,npoints);
		}

		if ( !PASSED(test_asymmetric_index_generator(n,npoints)) ) {
			fprintf(output,"Asymmetric index generator failed for n = %d and npoints = %d\n\n",n,npoints);
		}
		else {
			fprintf(output,"Asymmetric index generator worked for n = %d and npoints = %d\n\n",n,npoints);
		}
	}

	/* Permutation generators */
	if ( !test.CompareTo("permutations") ) {
		const int n = fx_param_int(NULL,"n",2);

		if ( !PASSED(test_permutation_generator(n)) ) {
			fprintf(output,"\n\nPermutation generator failed for n = %d\n\n", n);
		}
		else {
			fprintf(output,"\n\nPermutation generator worked for n = %d\n\n", n);
		}
	}
}

/*****************************************************************************/

success_t test_permutation_generator(int n) {
	DEBUG_ASSERT (n > 0);
	Vector tau;
	double count = 0.0;
	
	tau.Init(n);
	if ( !PASSED(generate_first_permutation(tau)) ) {
		return SUCCESS_FAIL;
	}
	count += 1.0;
	while ( PASSED(generate_next_permutation(tau)) ) {
		count += 1.0;
	}

	if ( count != math::Factorial(n) ) {
		return SUCCESS_FAIL;
	}

	return SUCCESS_PASS;
}

/*****************************************************************************/

success_t test_symmetric_index_generator(int n, index_t npoints) {
	double count = 1;
	Vector index;

	index.Init(n);
	if ( !PASSED(generate_new_index(index,npoints)) ) {
		return SUCCESS_WARN;
	}

	while ( PASSED(generate_next_symmetric_index(index,npoints)) ) {
		count += 1;
	}

	if (count != choose(npoints,n)) {
		return SUCCESS_FAIL;	
	}
	else {
		return SUCCESS_PASS;
	}
}


success_t test_asymmetric_index_generator(int n, index_t npoints) {
	double count = 1;
	Vector index;

	index.Init(n);
	if ( !PASSED(generate_new_index(index,npoints)) ) {
		return SUCCESS_WARN;
	}

	while ( PASSED(generate_next_asymmetric_index(index,npoints)) ) {
		count += 1;
	}

	if ( count != (choose(npoints,n) * math::Factorial(n)) ) {
		return SUCCESS_FAIL;	
	}
	else {
		return SUCCESS_PASS;
	}
}

/*****************************************************************************/

double choose(int n, int k) {
	double result = 1.0;
	int i;

	if (n == k || n == 0 || k == 0) {
		return result;
	}
	if (n < k) {
		int tmp = n;
		n = k;
		k = tmp;
	}

	for (i = 0; i < k; i++) {
		result *= (n-i);
	}
	result /= math::Factorial(k);

	return result;
}

/*****************************************************************************/


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

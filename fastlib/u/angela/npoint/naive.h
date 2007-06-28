/**
 * @author: Angela N Grigoroaia
 * @file: naive.h
 *
 * @description: Helper functions for the naive (base case) implementation.
 */

#ifndef NAIVE_H
#define NAIVE_H

#include "fastlib/fastlib.h"
#include "metrics.h"
#include "matcher.h"

/**
 * The return value for the following functions is a vector of results. The
 * first value represents the unweighted count and indicates that an error
 * occured if it is '-1'. If one or more weights were specified, the vector also
 * contains the weighted counts arranged in the same order in which the initial
 * weights were given.
 *
 * Example: 
 * 	Consider that 2 weights were given for each point. Then the results are:
 * 		[3, 10.4, 3.5]
 * 	where 3 is the unweighted count, 10.4 is the weighted count corresponding to
 * 	the first set of weights and 3.5 is the weighted count corresponding to the
 * 	second set of weights.
 */

/**
 * Wrapper function for n-aive npoint. This will select the appropriate function
 * to use by checking to see if we have a simple matcher and if we want to
 * caount all the permutations that match. The following cases are treated
 * separately:
 * 		- We want to count all permutations and we have a simple (symmetric)
 * 		matcher. In this case, if one permutation matches so do all the other n!
 * 		permutations. Thus, it is easier to use the simple symmetric n-point case
 * 		and	multiply the final result by n!. Note that this will also work for
 * 		weghted	results since the weighted contribution of an n-tuple is the same,
 * 		no matter what the order of the points is.
 * 		- We want to count all permutations and we have an asymmetric matcher. In
 * 		this case we really have to check all possible permutations so we use the
 * 		asymmetric function coupled with the simple matcher.
 * 		- We want to count an n-tuple only once, no matter how many permutations
 * 		actually match. We will use the symmetric version and for matching
 * 		permutations within the matcher.
 *
 * Note:
 * 	If we have an asymmetric matcher and we want to count regardless of how many
 * 	permutations actually match we will still call the symmetric version. The
 * 	matching procedure will try new permutations of the current n-tuple until it
 * 	finds a match or it runs out of possible permutations.
 * 	NOT PROPERLY IMPLEMENTED YET!!!
 */
Vector naive_npoint(DataPack data, Matcher matcher, Metric metric);

/**
 * This version only counts an n-tuple once even if more than one permutation of
 * the points might match. 
 *
 * Examples: 
 * 	1. Consider a symmetric matcher for 3-point correlation and a triangle ABC
 * 	that matches. Then ACB, BAC, BCA, CAB and CBA also match. However, since we
 * 	are not	interested in how many permutations match we will only count one
 * 	match.
 * 	2. Consider an asymmetric matcher for 3-point correlation and an isosceles
 * 	triangle ABC that matches. Assume that AB=AC. Then ACB will also match but
 * 	will not contribute to the count.
 */
Vector symmetric_naive_npoint(DataPack data, Matcher matcher, Metric metric);

/**
 * This version counts all the possible permutations that match. 
 *
 * Examples: 
 * 	1. Consider a symmetric matcher for 3-point correlation and a triangle ABC
 * 	that matches. Then ACB, BAC, BCA, CAB and CBA also match. Unlike the
 * 	previous function, all 6 permutations will contribute to the final count.
 * 	2. Consider an asymmetric matcher for 3-point correlation and an isosceles
 * 	triangle ABC that matches. Assume that AB=AC. Then ACB will also match and
 * 	contribute to the count. Also, the remaining permutations will also be
 * 	checked and, if they match they will also contribute to the count.
 */
Vector asymmetric_naive_npoint(DataPack data, Matcher matcher, Metric metric);

#endif


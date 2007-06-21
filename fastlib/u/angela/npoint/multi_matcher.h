/**
 * @author: Angela N Grigoroaia
 * @date: 15.05.2007
 * @file: multi_matcher.h
 *
 * Description: Useful functions for the multi-matcher problem.
 */

#ifndef MULTI_MATCHER_H
#define MULTI_MATCHER_H

#include "fastlib/fastlib.h"
#include "matcher.h"
#include "metrics.h"

/** 
 * Guess what the matcher interval should be and divide it into a suitable
 * number of matchers. This will create a list of matchers that we can use to
 * plot the 2-point correlation curve. We can use the list as a queue of
 * matchers over which we wil run naive or single tree n-point. We can also use
 * it for a divide & conquer approach (dual-tree, multi-matcher n-point).
 */
void estimate_multi_matcher_list(const Matrix data, const Metric metric) const;

/** 
 * Estimate the largest distance between any two points. This should be useful
 * for determining the maximum value for our matcher. We will use a 2-approx
 * algorithm:
 * 		1. Choose a random point x
 * 		2. Find y such that dist(x,y) is maximized
 * 		3. Find z such that dist(y,z) is maximized
 * 		4. Return dist (y,z)
 */
double estimate_diameter(const Matrix data, const Metric metric) const;

/** 
 * Given x, find y such that dist(x,y) is maximum. 
 * The function takes as input a matrix containing the entire dataset and an
 * integer corresponding to the index of x in the given matrix. It will return
 * the index of y.
 */
index_t find_farthest_neighbor(const Matrix data, const index_t x, 
		const Metric metric) const;


#endif

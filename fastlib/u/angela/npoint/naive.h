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
 * The return value is a vector of results. The first value represents the
 * unweighted count and indicates that an error occured if it is '-1'. If one or
 * more weights were specified, the vector also contains the weighted counts
 * arranged in the same order in which the initial weights were given.
 *
 * Example: 
 * 	Consider that 2 weights were given for each point. Then the results are:
 * 		[3, 10.4, 3.5]
 * 	where 3 is the unweighted count, 10.4 is the weighted count corresponding to
 * 	the first set of weights and 3.5 is the weighted count corresponding to the
 * 	second set of weights.
 */
Vector naive_npoint(DataPack data, Matcher matcher, Metric metric);

#endif


/**
 * @author: Angela N Grigoroaia
 * @date: 13.06.2007
 * @file: naive.h
 *
 * @description:
 * Helper functions for the naive (base case) implementation.
 */

#ifndef NAIVE_H
#define NAIVE_H

#include "fastlib/fastlib.h"
#include "metrics.h"
#include "matcher.h"

double naive_npoint(Matrix data, Matcher matcher, Metric metric);

#endif


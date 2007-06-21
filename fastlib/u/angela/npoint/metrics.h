/**
 * @author Angela N. Grigoroaia
 * @file metrics.h
 *
 * @description: Stuff that takes care of distance computations.
**/

/**
 * Requirements:
 * 	- Structure that stores the actual metric (norm) used
 *	- Functions that validate the fact that this is actually a metric (norm)
 *	- Functions that compute the distance between two points
 */

/** 
 * Metric description and considerations:
 *	- Stored as a class that contains a matrix and a flag. The flag triggers the
 *	use of a faster way of computing the distance for the euclidean metric. 
 *	- For any given vector v, its squared M-norm is given by 
 *			||v|| = v' * M * v 
 *	For example, the euclidian metric has M = I.
 *
 * Notice that any positive definite matrix will define a metric so this is the
 * only test we need to ansure we are using a valid metric. Since most of the
 * time we won't be needing fancy metrics, the default is the euclidean metric
 * with fast distance computation  enabled.
**/

#ifndef METRICS_H
#define METRICS_H

#include "fastlib/fastlib.h"
#include "globals.h"

class Metric {
	public:
		Matrix M;
		int dimension;
	  int fast_euclid;

	public:
		Metric() {}
		~Metric() {}

	/* This makes the default Euclidean metric with fast computation enabled. */
	public:
	void Init(const int size);

	/**
	 * This reads a metric from file. If the file cannot be accessed or there is a
	 * problem the default euclidean metric is used a a fallback. The following
	 * types of problems may appear:
	 * 	- The file is not accessible or does not contain a square matrix
	 * 	- The dimension of the metric stored in the file and the dimension
	 * 	specified as a parameter do not agree. The dimension parameter is
	 * 	considered to be the 'true' value and is used to create the fallback
	 * 	metric.
	 * 	- The stored matrix is not symmetric or positive-definite.
	 * 	If one of the sanity checks fails the return value is set to SUCCESS_WARN.
	 */
	public:
	success_t InitFromFile(const int size, const char *file);

	/* Distance computations */
	public:
	double ComputeDistance(const Vector a, const Vector b) const;
	double ComputeDistance(const Matrix data, const index_t x, const index_t y)
		const;
	double ComputeNorm(const Vector v) const;
};

/** 
 * Basic sanity check for the metric that only takes into account the actual
 * matrix.
 */
success_t is_positive_definite(const Matrix M);
success_t is_symmetric(const Matrix M);

/* Default distance computation for the Euclidean metric. */
double compute_fast_euclidean(const Vector a, const Vector b);

#endif

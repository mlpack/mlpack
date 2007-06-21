/**
 * @author: Angela N Grigoroaia
 * @file: multi_matcher.cc
 */

#include "multi_matcher.h"
#include "globals.h"
#include "fastlib/fastlib.h"
#include "metrics.h"
#include "matcher.h"
#include "naive.h"

void estimate_multi_matcher_list(const Matrix data, const Metric metric) {
	double est_diam = estimate_diameter(data,metric);
	}


double estimate_diameter(const Matrix data, const Metric metric) const {
	double diam = -1;
	index_t x,y,z;

	/** TODO: Choose x randomly instead of using the first point. */
	x = 0;
	y = find_farthest_neighbor(data,x,metric);
	z = find_farthest_neighbor(data,y,metric);

	diam = metric.ComputeDistance(data,y,z);
	return diam;
}


index_t find_farthest_neighbor(const Matrix data, const index_t x, 
		const Metric metric) const {
	double max_dist_so_far = -1;
	index_t result = -1, i;

	for (i=0;i<data.n_cols();i++) {
		double tmp_dist = metric.ComputeDistance(data,x,i);
		if (tmp_dist >= max_dist_so_far && x != i) {
			max_dist_so_far = tmp_dist;
			result = i;
		}
	}
	return result;
}

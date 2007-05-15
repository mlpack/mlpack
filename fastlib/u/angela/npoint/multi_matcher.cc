/**
 * @author: Angela N Grigoroaia
 * @date: 15.05.2007
 * @file: multi_matcher.cc
 *
 * Description: Generally useful function for the multi-matcher problem.
 */

#include "multi_matcher.h"

void estimate_multi_matcher_list(const Matrix data, const Metric metric) {
	double est_diam = estimate_diameter(data,metric);
	}


double estimate_diameter(const Matrix data, const Metric metric) {
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
		const Metric metric) {
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

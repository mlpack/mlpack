/*
 *  cuda_two_point.h
 *  
 *
 *  Created by William March on 10/3/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */


#ifndef CUDA_TWO_POINT_H
#define CUDA_TWO_POINT_H

#include <armadillo>
#include <iostream>
#include <string>
#include <boost/program_options.hpp>
#include "core/util/timer.h"
#include "core/tree/gen_metric_tree.h"
#include "mlpack/two_point/two_point_dev.h"

extern "C" void TwoPointKernelOnHost(
                                     double *query, int num_query_points,
                                     double *reference, int num_reference_points,
                                     int *two_point_sums_out, float lower_bound_sqr, float upper_bound_sqr);
namespace mlpack {
namespace cuda_two_point {

  void StartComputation(two_point::TwoPointArguments<core::table::Table <
                        core::tree::GenMetricTree <
                        mlpack::two_point::TwoPointStatistic> ,
                        mlpack::two_point::TwoPointResult > >& two_point_arguments,
                      mlpack::two_point::TwoPointResult* two_point_result) {
  
  // Tree type: hard-coded for a metric tree.
  typedef core::table::Table <
  core::tree::GenMetricTree <
  mlpack::two_point::TwoPointStatistic> ,
  mlpack::two_point::TwoPointResult > TableType;
  
  // Parse arguments for TwoPoint.
    /*
  mlpack::two_point::TwoPointArguments<TableType> two_point_arguments;
  if(mlpack::two_point::TwoPointArgumentParser::ParseArguments(vm, 
                                                               &two_point_arguments)) {
    return;
  }
     */
  
  
  // Invoke the CUDA kernel.
  int *two_point_sums_host =
  new int[ two_point_arguments.points_table_1_->n_entries()];
  
  double matcher_dist = two_point_arguments.matcher_distance_;
  double matcher_thick = two_point_arguments.matcher_thickness_;
  
  double lower_bound_sqr = (matcher_dist - 0.5*matcher_thick) 
  * (matcher_dist - 0.5*matcher_thick);
  double upper_bound_sqr = (matcher_dist + 0.5*matcher_thick) 
  * (matcher_dist + 0.5*matcher_thick);
  
  TwoPointKernelOnHost(two_point_arguments.points_table_1_->data().memptr(),
                       two_point_arguments.points_table_1_->n_entries(),
                       two_point_arguments.points_table_2_->data().memptr(),
                       two_point_arguments.points_table_2_->n_entries(),
                       two_point_sums_host, static_cast<float>(lower_bound_sqr), 
                       static_cast<float>(upper_bound_sqr));
  
  // Copy to the result. Avoid this by eliminating std::vector in the
  // TwoPointResult object.
  two_point_result->Init(two_point_arguments.points_table_1_->n_entries());
  
  // IMPORTANT: need to do a smarter reduction here
  for(int i = 0; i < two_point_arguments.points_table_1_->n_entries(); i++) {
    two_point_result->num_tuples_ += two_point_sums_host[i];
  }
    two_point_result->num_tuples_ = two_point_result->num_tuples_ / 2;
  
  printf("CUDA num_tuples: %d\n", two_point_result->num_tuples_);
  
}

} // namespace 
} // namespace

#endif 


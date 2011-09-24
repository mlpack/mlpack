/*
 *  two_point.cc
 *  
 *
 *  Created by William March on 9/12/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include <armadillo>
#include <iostream>
#include <string>
#include <boost/program_options.hpp>
#include "core/util/timer.h"
#include "core/tree/gen_metric_tree.h"
#include "mlpack/two_point/two_point_dev.h"
//#include "mlpack/series_expansion/kernel_aux.h"


void StartComputation(boost::program_options::variables_map& vm) {
  
  typedef core::table::Table <
    core::tree::GenMetricTree <mlpack::two_point::TwoPointStatistic>, 
    mlpack::two_point::TwoPointResult > TableType;
  
  mlpack::two_point::TwoPointArguments<TableType> arguments;
  if(mlpack::two_point::TwoPointArgumentParser::ParseArguments(vm, &arguments)) 
  {
    return;
  }

  // Initialize the algorithm class
  core::util::Timer init_timer;
  init_timer.Start();
  mlpack::two_point::TwoPoint<TableType> alg;
  alg.Init(arguments,
         (typename mlpack::two_point::TwoPoint<TableType>::GlobalType *) NULL);
  // alg.Init(arguments, (void *)NULL);
  init_timer.End();
  printf("%g seconds elapsed in initializing...\n",
         init_timer.GetTotalElapsedTime());
  
  // Compute the result.
  core::util::Timer compute_timer;
  compute_timer.Start();
  mlpack::two_point::TwoPointResult result;
  alg.Compute(arguments, &result);
  compute_timer.End();
  printf("%g seconds elapsed in computation...\n",
         compute_timer.GetTotalElapsedTime());
  
  
  // Output results here
  
  
} // StartComputation()


int main(int argc, char* argv[]) { 

  boost::program_options::variables_map vm;
  if (mlpack::two_point::TwoPointArgumentParser::ConstructBoostVariableMap(
          argc, argv, &vm)) {
    
    return 1;
    
  } // check if we parsed the arguments
  
  
  StartComputation(vm);
  
  return 0;
  
} // main

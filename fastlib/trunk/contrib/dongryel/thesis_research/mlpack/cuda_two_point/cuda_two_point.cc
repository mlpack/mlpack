/** @file cuda_two_point.cc
 *
 *  The main driver for the CUDA based two point.
 *
 *  @author Bill March (march@gatech.edu)
 */

#include "cuda_two_point.h"
#include "mlpack/two_point/two_point_result.h"

int main(int argc, char *argv[]) {

  typedef core::table::Table <core::tree::GenMetricTree <
  mlpack::two_point::TwoPointStatistic> ,
  mlpack::two_point::TwoPointResult > TableType;
  
  
  boost::program_options::variables_map vm;
  if(mlpack::two_point::TwoPointArgumentParser::ConstructBoostVariableMap(
        argc, argv, &vm)) {
    return 1;
  }
  
  mlpack::two_point::TwoPointArguments<TableType> two_point_arguments;
  
  
  if(mlpack::two_point::TwoPointArgumentParser::ParseArguments(vm, 
                                                   &two_point_arguments)) {
    return 1;
  }
  

  mlpack::two_point::TwoPointResult result;
  
  mlpack::cuda_two_point::StartComputation(two_point_arguments, &result);
  
  printf("CUDA num tuples: %d\n", result.num_tuples_);
  
  return 0;
}

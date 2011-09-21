/*
 *  distributed_two_point.cc
 *  
 *
 *  Created by William March on 9/20/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "core/metric_kernels/lmetric.h"
#include "core/tree/statistic.h"
#include "core/table/distributed_table.h"
#include "core/tree/gen_kdtree.h"
#include "core/tree/gen_metric_tree.h"
#include "core/math/math_lib.h"
#include "mlpack/two_point/two_point_dualtree.h"
#include "mlpack/distributed_two_point/distributed_two_point_dev.h"

void StartComputation(boost::mpi::communicator &world,
                      boost::program_options::variables_map &vm) {
  
  // Tree type: hard-coded for a metric tree.
  typedef core::table::DistributedTable <
  core::tree::GenKdTree <mlpack::two_point::TwoPointStatistic>,
  mlpack::two_point::TwoPointResult > DistributedTableType;
  
  // Parse arguments for Kde.
  mlpack::distributed_two_point::DistributedTwoPointArguments <
  DistributedTableType > distributed_two_point_arguments;
  if(mlpack::distributed_two_point::
     DistributedTwoPointArgumentParser::ParseArguments(
                            world, vm, &distributed_two_point_arguments)) {
       return;
     }
  
  // Instantiate a distributed KDE object.
  mlpack::distributed_two_point::DistributedTwoPoint <
    DistributedTableType> distributed_two_point_instance;
  distributed_two_point_instance.Init(world, distributed_two_point_arguments);
  
  // Compute the result.
  mlpack::two_point::TwoPointResult two_point_result;
  distributed_two_point_instance.Compute(
                         distributed_two_point_arguments, &two_point_result);
  
  // Output the KDE result to the file.
  //std::cerr << "Writing the densities to the file: " <<
  //distributed_kde_arguments.densities_out_ << "\n";
  //kde_result.Print(distributed_kde_arguments.densities_out_);


} // StartComputation

int main(int argc, char *argv[]) {
  
  // Initialize boost MPI.
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
  
  // Seed the random number.
  core::math::global_random_number_state_.set_seed(time(NULL) + world.rank());
  
  if(world.rank() == 0) {
    printf("%d processes are present...\n", world.size());
  }
  
  boost::program_options::variables_map vm;
  if(mlpack::distributed_two_point::
     DistributedTwoPointArgumentParser::ConstructBoostVariableMap(
                                               world, argc, argv, &vm)) {
       return 0;
   }
  
  StartComputation(world, vm);

  return 0;
}





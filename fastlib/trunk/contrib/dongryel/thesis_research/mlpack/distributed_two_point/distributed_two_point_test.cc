/*
 *  distributed_two_point_test.cc
 *  
 *
 *  Created by William March on 9/22/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */


#include "core/tree/gen_metric_tree.h"
#include "mlpack/two_point/test_two_point.h"
#include "mlpack/distributed_two_point/distributed_two_point_dev.h"
//#include "mlpack/series_expansion/kernel_aux.h"
#include <time.h>

namespace mlpack {
  namespace distributed_two_point {
    
    /** @brief The test driver for the distributed two point.
     */
    class TestDistributedTwoPoint {
      
    private:
      
      template<typename TableType>
      void CopyTable_(TableType *local_table, double **start_ptrs) {
        int n_attributes = local_table->n_attributes();
        
        // Look at old_from_new_indices.
        std::pair<int, std::pair<int, int> > *old_from_new =
        local_table->old_from_new();
        
        for(int i = 0; i < local_table->n_entries(); i++) {
          arma::vec point;
          int old_process_index = old_from_new[i].first;
          int old_process_point_index = old_from_new[i].second.first;
          int new_process_old_from_new_index =
          old_from_new[i].second.second;
          local_table->get(new_process_old_from_new_index, &point);
          double *destination = start_ptrs[old_process_index] +
          n_attributes * old_process_point_index;
          memcpy(destination, point.memptr(), sizeof(double) * n_attributes);
        }
      }
      
      template<typename DistributedTableType, typename TableType>
      void CombineTables_(boost::mpi::communicator &world,
                          DistributedTableType *distributed_reference_table,
                          TableType &output_table, int **total_distribution_in) {
        int total_num_points = 0;
        TableType *local_table = distributed_reference_table->local_table();
        int n_attributes = local_table->n_attributes();
        double **start_ptrs = NULL;
        
        // The master process needs to figure out the layout of the
        // original tables.
        int *point_distribution = new int[world.size()];
        *total_distribution_in = new int[world.size()];
        int *total_distribution = *total_distribution_in;
        for(int i = 0; i < world.size(); i++) {
          point_distribution[i] = total_distribution[i] = 0;
        }
        std::pair<int, std::pair<int, int> > *old_from_new =
        local_table->old_from_new();
        for(int i = 0; i < local_table->n_entries(); i++) {
          point_distribution[ old_from_new[i].first ]++;
        }
        boost::mpi::all_reduce(world, point_distribution, world.size(),
                               total_distribution, std::plus<int>());
        
        // The master process initializes the global table and the
        // copies its own data onto it.
        if(world.rank() == 0) {
          start_ptrs = new double *[world.size()];
          for(int i = 0; i < world.size(); i++) {
            total_num_points += total_distribution[i];
          }
          output_table.Init(n_attributes, total_num_points);
          start_ptrs[0] = output_table.data().memptr();
          total_num_points = 0;
          for(int i = 0; i < world.size(); i++) {
            total_num_points += total_distribution[i];
            if(i + 1 < world.size()) {
              start_ptrs[i + 1] =
              start_ptrs[0] + total_num_points * n_attributes;
            }
          }
          CopyTable_(local_table, start_ptrs);
        }
        
        for(int i = 1; i < world.size(); i++) {
          TableType received_table;
          if(world.rank() == 0) {
            // Receive the table from $i$-th process and copy.
            world.recv(i, boost::mpi::any_tag, received_table);
            CopyTable_(&received_table, start_ptrs);
          }
          else {
            // Send the table to the master process.
            world.send(0, i, *local_table);
            break;
          }
        }
        if(world.rank() == 0) {
          delete[] start_ptrs;
        }
        delete[] point_distribution;
      }
 
      
    public:
      
      int StressTestMain(boost::mpi::communicator &world) {
        for(int i = 0; i < 40; i++) {
          
          StressTest(world);
          
        }
        return 0;
      }
      
      int StressTest(boost::mpi::communicator &world) {
        
        // Typedef the trees and tables.
        typedef core::tree::GenMetricTree <
        mlpack::two_point::TwoPointStatistic> TreeSpecType;
        typedef core::table::DistributedTable <
        TreeSpecType, mlpack::two_point::TwoPointResult > DistributedTableType;
        typedef DistributedTableType::TableType TableType;
        
        // Only the master generates the number of dimensions.
        int num_dimensions;
        if(world.rank() == 0) {
          num_dimensions = core::math::RandInt(2, 4);
        }
        boost::mpi::broadcast(world, num_dimensions, 0);
        //int num_points = core::math::RandInt(300, 500);
        int num_points = core::math::RandInt(3, 10);
        std::vector< std::string > args;
        
        // Push in the random generate command.
        std::stringstream random_generate_n_attributes_sstr;
        random_generate_n_attributes_sstr << "--random_generate_n_attributes=" <<
        num_dimensions;
        args.push_back(random_generate_n_attributes_sstr.str());
        std::stringstream random_generate_n_entries_sstr;
        random_generate_n_entries_sstr << "--random_generate_n_entries=" <<
        num_points;
        args.push_back(random_generate_n_entries_sstr.str());
        args.push_back("--random_generate");
        
        //args.push_back("--num_threads_in=2");
        
        // Push in the reference dataset name.
        std::string data_in("random_dataset.csv");
        args.push_back(std::string("--data_in=") + data_in);
        
        // Push in the densities output file name.
        args.push_back(std::string("--counts_out=counts.txt"));
        
        // Push in the kernel type.
        if(world.rank() == 0) {
          std::cout << "\n==================\n";
          std::cout << "Test trial begin\n";
          std::cout << "Number of dimensions: " << num_dimensions << "\n";
          fflush(stdout);
          fflush(stderr);
        }
        std::cout << "Number of points generated by " <<
        world.rank() << ": " << num_points << "\n";
        
        
        // Push in the leaf size.
        int leaf_size = 0;
        if(world.rank() == 0) {
          //leaf_size = core::math::RandInt(15, 25);
          leaf_size = 100;
        }
        boost::mpi::broadcast(world, leaf_size, 0);
        std::stringstream leaf_size_sstr;
        leaf_size_sstr << "--leaf_size=" << leaf_size;
        args.push_back(leaf_size_sstr.str());
        
        double matcher_distance;
        if (world.rank() == 0) {
          matcher_distance =
          core::math::Random(0.1 * sqrt(num_dimensions),
                             0.5 * sqrt(num_dimensions));
        }
        boost::mpi::broadcast(world, matcher_distance, 0);
        std::stringstream matcher_distance_sstr;
        matcher_distance_sstr << "--matcher_distance=" << matcher_distance;
        args.push_back(matcher_distance_sstr.str());
        
        // Push in the randomly generated matcher_distance.
        double matcher_thickness;
        
        if (world.rank() == 0) {
        matcher_thickness =
          core::math::Random(0.01 * sqrt(num_dimensions),
                             0.025 * sqrt(num_dimensions));
        }
        boost::mpi::broadcast(world, matcher_thickness, 0);
        std::stringstream matcher_thickness_sstr;
        matcher_thickness_sstr << "--matcher_thickness=" << matcher_thickness;
        args.push_back(matcher_thickness_sstr.str());
        
        /*
        int num_threads = core::math::RandInt(1, 5);
        std::stringstream num_threads_sstr;
        num_threads_sstr << "--num_threads_in=" << num_threads;
        args.push_back(num_threads_sstr.str());
        */
        
        // Push in the randomly generate work parameters.
        double max_subtree_size;
        double max_num_work_to_dequeue_per_stage;
        if(world.rank() == 0) {
          max_subtree_size = core::math::RandInt(60, 200);
          max_num_work_to_dequeue_per_stage = core::math::RandInt(3, 10);
        }
        boost::mpi::broadcast(world, max_subtree_size, 0);
        boost::mpi::broadcast(world, max_num_work_to_dequeue_per_stage, 0);
        std::stringstream max_subtree_size_sstr;
        std::stringstream max_num_work_to_dequeue_per_stage_sstr;
        max_subtree_size_sstr
        << "--max_subtree_size_in=" << max_subtree_size;
        max_num_work_to_dequeue_per_stage_sstr
        << "--max_num_work_to_dequeue_per_stage_in=" <<
        max_num_work_to_dequeue_per_stage;
        args.push_back(max_subtree_size_sstr.str());
        args.push_back(max_num_work_to_dequeue_per_stage_sstr.str());
        
        // Parse the distributed  arguments.
        mlpack::distributed_two_point::DistributedTwoPointArguments <
        DistributedTableType > distributed_two_point_arguments;
        boost::program_options::variables_map vm;
        mlpack::distributed_two_point::DistributedTwoPointArgumentParser::
        ConstructBoostVariableMap(world, args, &vm);
        mlpack::distributed_two_point::DistributedTwoPointArgumentParser::ParseArguments(
                  world, vm, &distributed_two_point_arguments);
        
        if(world.rank() == 0) {
          std::cout << "Matcher distance " << matcher_distance << "\n";
          std::cout << "Matcher thickness " << matcher_thickness << "\n";
        }
        world.barrier();
        
        // Call the distributed driver.
        mlpack::distributed_two_point::DistributedTwoPoint <
        DistributedTableType> distributed_two_point_instance;
        distributed_two_point_instance.Init(world, distributed_two_point_arguments);
        
        // Compute the result.
        mlpack::two_point::TwoPointResult distributed_two_point_result;
        distributed_two_point_instance.Compute(
                                         distributed_two_point_arguments,
                                               &distributed_two_point_result);
        
        // For each process, check whether all the othe reference points
        // have been encountered.
        DistributedTableType *distributed_reference_table =
        distributed_two_point_arguments.points_table_1_;
                
        // Call the ultra-naive.
        int ultra_naive_distributed_two_point_result;
        
        // The master collects all the distributed tables and collects a
        // mega-table for which can be used to compute the naive
        // results.
        TableType combined_reference_table;
        int *total_distribution;
        CombineTables_(world, distributed_reference_table, combined_reference_table,
                       &total_distribution);
        
        
        // TODO: IMPORTANT: make sure this is reading from the file too
        if(world.rank() == 0) {
          printf("naive num points: %d\n", combined_reference_table.n_entries());
          
          double naive_match = distributed_two_point_instance.global().matcher_distance();
          double naive_thick = distributed_two_point_instance.global().matcher_thickness();
          ultra_naive_distributed_two_point_result = mlpack::two_point::TestTwoPoint::UltraNaive(
                           *(distributed_two_point_arguments.metric_),
                           combined_reference_table, combined_reference_table,
                           true, naive_match, naive_thick);
        }
        
        // The master broadcasts the ultranaive result to all processes,
        // each of which checks against it.
        boost::mpi::broadcast(world, ultra_naive_distributed_two_point_result, 
                              0);
        
        
        printf("naive tuples: %d, distributed tuples: %d\n",
               ultra_naive_distributed_two_point_result,
               distributed_two_point_result.num_tuples_);
        if(distributed_two_point_result.num_tuples_ !=
                          ultra_naive_distributed_two_point_result) {
          std::cerr << "There is a problem!\n";
          if (world.rank() == 0) {
            std::string filename("problem_case.csv");
            combined_reference_table.Save(filename);
          }
          exit(-1);
        }
        
        world.barrier();
        
        return 0;
      }
    };
  }
}

int main(int argc, char *argv[]) {
  
  // Initialize boost MPI.
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
  
  // Delete the teporary files and put a barrier.
  std::stringstream temporary_file_name;
  temporary_file_name << "tmp_file" << world.rank();
  remove(temporary_file_name.str().c_str());
  world.barrier();
  core::math::global_random_number_state_.set_seed(time(NULL) + world.rank());
  
  // Call the tests.
  mlpack::distributed_two_point::TestDistributedTwoPoint distributed_two_point_test;
  distributed_two_point_test.StressTestMain(world);
  
  if(world.rank() == 0) {
    std::cout << "All tests passed!\n";
  }
  fflush(stdout);
  fflush(stderr);
  world.barrier();
  return 0;
}

/*
 *  distributed_two_point_dev.h
 *  
 *
 *  Created by William March on 9/20/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef MLPACK_DISTRIBUTED_TWO_POINT_DEV_H
#define MLPACK_DISTRIBUTED_TWO_POINT_DEV_H

#include <omp.h>
#include "core/parallel/distributed_dualtree_dfs_dev.h"
#include "core/parallel/random_dataset_generator.h"
#include "core/metric_kernels/lmetric.h"
#include "core/table/memory_mapped_file.h"
#include "core/table/transform.h"
#include "mlpack/distributed_two_point/distributed_two_point.h"
#include "boost/mpi/collectives.hpp"

namespace core {
  namespace table {
    extern core::table::MemoryMappedFile *global_m_file_;
  }
}

namespace mlpack {
  namespace distributed_two_point {

    template<typename DistributedTableType>
    void DistributedTwoPoint<DistributedTableType>::Compute(
              const mlpack::distributed_two_point::DistributedTwoPointArguments <
              DistributedTableType > &arguments_in,
              ResultType *result_out) {

      // Barrier so that every process is here.
      world_->barrier();
      
      // Instantiate a dual-tree algorithm of the TwoPoint.
      core::parallel::DistributedDualtreeDfs <
        mlpack::distributed_two_point::DistributedTwoPoint <
        DistributedTableType> >
      distributed_dualtree_dfs;
      distributed_dualtree_dfs.Init(world_, *this);
      distributed_dualtree_dfs.set_work_params(
         arguments_in.leaf_size_,
         arguments_in.max_subtree_size_,
         arguments_in.do_load_balancing_,
         arguments_in.max_num_work_to_dequeue_per_stage_);

      // Compute the result and do post-normalize.
      distributed_dualtree_dfs.Compute(* arguments_in.metric_, result_out);
      //result_out->Normalize(global_);
      
      printf("process: %d, Number of tuples: %d\n", world_->rank(),
             result_out->num_tuples_);

      int total_num_tuples;
      boost::mpi::all_reduce(*world_, result_out->num_tuples_, total_num_tuples, 
                             std::plus<int>());
      
      printf("total_num_tuples: %d\n", total_num_tuples);
      
      result_out->num_tuples_ = total_num_tuples;
      
    } // Compute()
    
    
    
    template<typename DistributedTableType>
    void DistributedTwoPoint<DistributedTableType>::Init(
               boost::mpi::communicator &world_in,
               mlpack::distributed_two_point::DistributedTwoPointArguments <
               DistributedTableType > &arguments_in) {

      world_ = &world_in;
      points_table_1_ = arguments_in.points_table_1_;
      if(arguments_in.points_table_2_ == NULL) {
        //printf("is mono\n");
        is_monochromatic_ = true;
        points_table_2_ = points_table_1_;
      }
      else {
        //printf("not mono\n");
        is_monochromatic_ = false;
        points_table_2_ = arguments_in.points_table_2_;
      }
      
      // Declare the global constants.
      global_.Init(points_table_1_, points_table_2_, 
                   arguments_in.matcher_distance_,
                   arguments_in.matcher_thickness_,
                   is_monochromatic_);
      
      } // Init()
  
    
  bool DistributedTwoPointArgumentParser::ConstructBoostVariableMap(
  boost::mpi::communicator &world,
  const std::vector<std::string> &args,
  boost::program_options::variables_map *vm) {

  boost::program_options::options_description desc("Available options");
  desc.add_options()(
    "help", "Print this information."
  )(
    "counts_out",
    boost::program_options::value<std::string>()->default_value(
      "counts_out.csv"),
    "OPTIONAL file to store computed counts."
  )(
    "do_load_balancing",
    "If present, do load balancing."
  )(
    "leaf_size",
    boost::program_options::value<int>()->default_value(40),
    "Maximum number of points at a leaf of the tree."
  )(
    "max_num_work_to_dequeue_per_stage_in",
    boost::program_options::value<int>()->default_value(5),
    "The number of work items to dequeue per process."
  )(
    "max_subtree_size_in",
    boost::program_options::value<int>()->default_value(20000),
    "The maximum size of the subtree to serialize at a given moment."
  )(
    "memory_mapped_file_size",
    boost::program_options::value<unsigned int>(),
    "The size of the memory mapped file."
  )(
    "num_threads_in",
    boost::program_options::value<int>()->default_value(1),
    "The number of threads to use for shared-memory parallelism."
  )(
    "data_in",
    boost::program_options::value<std::string>(),
    "REQUIRED file containing the point positions."
  )(
    "randoms_in",
    boost::program_options::value<std::string>(),
    "OPTIONAL file containing random (Poisson) positions.  If omitted, "
    "then the DD count is computed."
  )(
     "matcher_distance",
     boost::program_options::value<double>(),
     "REQUIRED The scale of the matcher (r)."
  )(
     "matcher_thickness",
     boost::program_options::value<double>(),
     "REQUIRED The width of the matcher (Delta r)"
  )(
    "random_generate",
    "If present, generate the datasets on the fly."
  )(
    "random_generate_n_attributes",
    boost::program_options::value<int>()->default_value(5),
    "Generate the datasets on the fly of the specified dimension."
  )(
    "random_generate_n_entries",
    boost::program_options::value<int>()->default_value(100000),
    "Generate the datasets on the fly of the specified number of points."
  )( 
    "random_seed_in",
    boost::program_options::value<unsigned long int>(),
    "Random seed to start for each MPI process."
  )(
    "top_tree_sample_probability",
    boost::program_options::value<double>()->default_value(0.3),
    "The portion of points sampled on each MPI process for building the "
    "top tree."
  )(
    "use_memory_mapped_file",
    "Use memory mapped file for out-of-core computations."
  );

    boost::program_options::command_line_parser clp(args);
    clp.style(boost::program_options::command_line_style::default_style
              ^ boost::program_options::command_line_style::allow_guessing);
    try {
      boost::program_options::store(clp.options(desc).run(), *vm);
    }
    catch(const boost::program_options::invalid_option_value &e) {
      std::cerr << "Invalid Argument: " << e.what() << "\n";
      exit(0);
    }
    catch(const boost::program_options::invalid_command_line_syntax &e) {
      std::cerr << "Invalid command line syntax: " << e.what() << "\n";
      exit(0);
    }
    catch(const boost::program_options::unknown_option &e) {
      std::cerr << "Unknown option: " << e.what() << "\n";
      exit(0);
    }
    
    boost::program_options::notify(*vm);
    if(vm->count("help")) {
      std::cout << desc << "\n";
      return true;
    }
    
    if(vm->count("data_in") == 0) {
      std::cerr << "Missing required --data_in.\n";
      exit(0);
    }
    if((*vm)["leaf_size"].as<int>() <= 0) {
      std::cerr << "The --leaf_size needs to be a positive integer.\n";
      exit(0);
    }
    if ((*vm)["matcher_distance"].as<double>() <= 0.0) {
      std::cerr << "The --matcher_distance needs to be a positive real.\n";
      exit(0);    
    }
    if ((*vm)["matcher_thickness"].as<double>() <= 0.0) {
      std::cerr << "The --matcher_thickness needs to be a positive real.\n";
      exit(0);        
    }
    
    if((*vm)["max_num_work_to_dequeue_per_stage_in"].as<int>() <= 0) {
      std::cerr << "The --max_num_work_to_dequeue_per_stage_in needs to be " <<
      "a positive integer.\n";
      exit(0);
    }
    if((*vm)["max_subtree_size_in"].as<int>() <= 1) {
      std::cerr << "The --max_subtree_size_in needs to be " <<
      "a positive integer greater than 1.\n";
      exit(0);
    }
    if((*vm)["num_threads_in"].as<int>() <= 0) {
      std::cerr << "The --num_threads_in needs to be a positive integer.\n";
      exit(0);
    }
    if(vm->count("random_generate_n_attributes") > 0) {
      if(vm->count("random_generate_n_entries") == 0) {
        std::cerr << "Missing required --random_generate_n_entries.\n";
        exit(0);
      }
      if((*vm)["random_generate_n_attributes"].as<int>() <= 0) {
        std::cerr << "The --random_generate_n_attributes requires a positive "
        "integer.\n";
        exit(0);
      }
    }
    if(vm->count("random_generate_n_entries") > 0) {
      if(vm->count("random_generate_n_attributes") == 0) {
        std::cerr << "Missing required --random_generate_n_attributes.\n";
        exit(0);
      }
      if((*vm)["random_generate_n_entries"].as<int>() <= 0) {
        std::cerr << "The --random_generate_n_entries requires a positive "
        "integer.\n";
        exit(0);
      }
    }
    
    if(vm->count("use_memory_mapped_file") > 0) {
      
      if(vm->count("memory_mapped_file_size") == 0) {
        std::cerr << "The --used_memory_mapped_file requires an additional "
        "parameter --memory_mapped_file_size.\n";
        exit(0);
      }
      unsigned int memory_mapped_file_size =
      (*vm)["memory_mapped_file_size"].as<unsigned int>();
      if(memory_mapped_file_size <= 0) {
        std::cerr << "The --memory_mapped_file_size needs to be a positive "
        "integer.\n";
        exit(0);
      }
      
      // Delete the teporary files and put a barrier.
      std::stringstream temporary_file_name;
      temporary_file_name << "tmp_file" << world.rank();
      remove(temporary_file_name.str().c_str());
      world.barrier();
      
      // Initialize the memory allocator.
      core::table::global_m_file_ = new core::table::MemoryMappedFile();
      core::table::global_m_file_->Init(
            std::string("tmp_file"), world.rank(), world.rank(), 100000000);
    }
    
    return false;
  } // ConstructBoostVariableMap
    
    
    template<typename DistributedTableType>
    bool DistributedTwoPointArgumentParser::ParseArguments(
            boost::mpi::communicator &world,
            boost::program_options::variables_map &vm,
            mlpack::distributed_two_point::DistributedTwoPointArguments <
                                     DistributedTableType > *arguments_out) {

      // Define the table type.
      typedef typename DistributedTableType::TableType TableType;
      
      // A L2 metric to index the table to use.
      arguments_out->metric_ = new core::metric_kernels::LMetric<2>();
      
      // Parse the load balancing option.
      arguments_out->do_load_balancing_ = (vm.count("do_load_balancing") > 0);
      
      // Parse the top tree sample probability.
      arguments_out->top_tree_sample_probability_ =
      vm["top_tree_sample_probability"].as<double>();
      if(world.rank() == 0) {
        std::cout << "Sampling the number of points owned by each MPI process with "
        "the probability of " <<
        arguments_out->top_tree_sample_probability_ << "\n";
      }
      
      // Parse the densities out file.
      arguments_out->counts_out_ = vm["counts_out"].as<std::string>();
      if(vm.count("random_generate_n_entries") > 0) {
        std::stringstream counts_out_sstr;
        counts_out_sstr << vm["counts_out"].as<std::string>() <<
        world.rank();
        arguments_out->counts_out_ = counts_out_sstr.str();
      }
      
      // Parse the leaf size.
      arguments_out->leaf_size_ = vm["leaf_size"].as<int>();
      if(world.rank() == 0) {
        std::cout << "Using the leaf size of " << arguments_out->leaf_size_ << "\n";
      }
      
      // Parse the number of threads.
      arguments_out->num_threads_ = vm["num_threads_in"].as<int>();
      omp_set_num_threads(arguments_out->num_threads_);
      //printf("arguments_out->num_threads_ = %d\n", arguments_out->num_threads_);
      std::cerr << "  Process " << world.rank() << " is using " <<
      arguments_out->num_threads_ << " threads for " <<
        "shared memory parallelism.\n";
      
      // Parse the random seed if it is available and set it.
      if(vm.count("random_seed_in") > 0) {
        unsigned long int seed = vm["random_seed_in"].as<unsigned long int>();
        core::math::global_random_number_state_.set_seed(seed + world.rank());
      }
      
      // Parse the data set and index the tree.
      std::string data_file_name = vm["data_in"].as<std::string>();
      arguments_out->points_table_1_ =
      (core::table::global_m_file_) ?
      core::table::global_m_file_->Construct<DistributedTableType>() :
      new DistributedTableType();
      
      if(vm.count("random_generate") > 0) {
        std::stringstream data_file_name_sstr;
        data_file_name_sstr << vm["data_in"].as<std::string>() <<
        world.rank();
        data_file_name = data_file_name_sstr.str();
        TableType *random_data_dataset =
        (core::table::global_m_file_) ?
        core::table::global_m_file_->Construct<TableType>() : new TableType();
        //core::parallel::RandomDatasetGenerator::Generate(
        //     vm["random_generate_n_attributes"].as<int>(),
        //     vm["random_generate_n_entries"].as<int>(), world.rank(),
         //    vm["prescale"].as<std::string>(), false, random_data_dataset);
        core::parallel::RandomDatasetGenerator::Generate(
                                                         vm["random_generate_n_attributes"].as<int>(),
                                                         vm["random_generate_n_entries"].as<int>(), world.rank(),
        //                                                 vm["prescale"].as<std::string>(), 
                                                         std::string("none"),
                                                         false, random_data_dataset);
        arguments_out->points_table_1_->Init(random_data_dataset, world);
      }
      else {
        std::cout << "Reading in the data set: " <<
        data_file_name << "\n";
        arguments_out->points_table_1_->Init(data_file_name, world);
        std::cout << "Finished reading in the data set.\n";
      }
      // Chromaticity 0
      std::cout << "Building the data tree.\n";
      arguments_out->points_table_1_->IndexData(
          *(arguments_out->metric_), world, arguments_out->leaf_size_,
           arguments_out->top_tree_sample_probability_, 0);
      std::cout << "Finished building the data tree.\n";
      
      // Parse the random set and index the tree.
      if(vm.count("randoms_in") > 0) {
        std::string randoms_file_name = vm["randoms_in"].as<std::string>();
        arguments_out->points_table_2_ =
        (core::table::global_m_file_) ?
        core::table::global_m_file_->Construct<DistributedTableType>() :
        new DistributedTableType();
        
        std::cout << "Reading in the random set: " <<
        randoms_file_name << "\n";
        arguments_out->points_table_2_->Init(randoms_file_name, world);
        std::cout << "Finished reading in the random set.\n";
        
        // Chromaticity 1
        std::cout << "Building the random tree.\n";
        arguments_out->points_table_2_->IndexData(
               *(arguments_out->metric_), world, arguments_out->leaf_size_,
               arguments_out->top_tree_sample_probability_, 1);

        std::cout << "Finished building the random tree.\n";
      }
      
      // Parse the bandwidth.
      arguments_out->matcher_distance_ = vm["matcher_distance"].as<double>();
      std::cout << "Matcher distance of " << arguments_out->matcher_distance_ << "\n";
      
      arguments_out->matcher_thickness_ = vm["matcher_thickness"].as<double>();
      std::cout << "Matcher thickness of " << arguments_out->matcher_thickness_ << "\n";

      // Parse the work parameters for the distributed engine.
      arguments_out->max_subtree_size_ =
      std::min(
               vm["max_subtree_size_in"].as<int>(),
               arguments_out->points_table_1_->n_entries() /
               arguments_out->num_threads_);
      arguments_out->max_num_work_to_dequeue_per_stage_ =
      vm["max_num_work_to_dequeue_per_stage_in"].as<int>();
      if(world.rank() == 0) {
        std::cout << "Serializing " << arguments_out->max_subtree_size_
        << " points of the tree at a time.\n";
        std::cout << "Dequeuing " <<
        arguments_out->max_num_work_to_dequeue_per_stage_ <<
        " items at a time from each process.\n";
      }
      
      return false;
    }
    
    
    bool DistributedTwoPointArgumentParser::ConstructBoostVariableMap(
                   boost::mpi::communicator &world,
                   int argc,
                   char *argv[],
                   boost::program_options::variables_map *vm) {

      // Convert C input to C++; skip executable name for Boost.
      std::vector<std::string> args(argv + 1, argv + argc);
      
      return ConstructBoostVariableMap(world, args, vm);
    }
    
    

  }// namespace
} // namespace
    
#endif

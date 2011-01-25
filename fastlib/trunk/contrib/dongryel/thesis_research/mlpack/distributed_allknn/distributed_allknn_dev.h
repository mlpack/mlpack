/** @file distributed_allknn_dev.h
 *
 *  The driver file for the distributed implementation of all
 *  k-nearest neighbors.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_DISTRIBUTED_ALLKNN_DISTRIBUTED_ALLKNN_DEV_H
#define MLPACK_DISTRIBUTED_ALLKNN_DISTRIBUTED_ALLKNN_DEV_H

#include "core/gnp/distributed_dualtree_dfs_dev.h"
#include "core/metric_kernels/lmetric.h"
#include "core/table/memory_mapped_file.h"
#include "mlpack/distributed_allknn/distributed_allknn.h"

namespace core {
namespace table {
extern core::table::MemoryMappedFile *global_m_file_;
}
}

namespace mlpack {
namespace distributed_allknn {

template<typename DistributedTableType>
DistributedTableType *DistributedAllknn<DistributedTableType>::query_table() {
  return query_table_;
}

template<typename DistributedTableType>
DistributedTableType *DistributedAllknn<DistributedTableType>::reference_table() {
  return reference_table_;
}

template<typename DistributedTableType>
typename DistributedAllknn<DistributedTableType>::GlobalType &
DistributedAllknn<DistributedTableType>::global() {
  return global_;
}

template<typename DistributedTableType>
bool DistributedAllknn<DistributedTableType>::is_monochromatic() const {
  return is_monochromatic_;
}

template<typename DistributedTableType>
void DistributedAllknn<DistributedTableType>::Compute(
  const mlpack::distributed_allknn::DistributedAllknnArguments <
  DistributedTableType > &arguments_in,
  mlpack::allknn::AllknnResult< std::vector<double> > *result_out) {

  // Barrier so that every process is here.
  world_->barrier();

  boost::mpi::timer timer;

  // Instantiate a dual-tree algorithm of the allknn.
  core::gnp::DistributedDualtreeDfs <
  mlpack::distributed_allknn::DistributedAllknn<DistributedTableType> >
  distributed_dualtree_dfs;
  distributed_dualtree_dfs.Init(world_, *this);
  distributed_dualtree_dfs.set_work_params(
    arguments_in.max_num_levels_to_serialize_,
    arguments_in.max_num_work_to_dequeue_per_stage_);

  // Compute the result and do post-normalize.
  distributed_dualtree_dfs.Compute(* arguments_in.metric_, result_out);
  result_out->Normalize(global_);

  if(world_->rank() == 0) {
    printf("Spent %g seconds in computation.\n", timer.elapsed());
  }
}

template<typename DistributedTableType>
void DistributedAllknn<DistributedTableType>::Init(
  boost::mpi::communicator &world_in,
  mlpack::distributed_allknn::DistributedAllknnArguments <
  DistributedTableType > &arguments_in) {

  world_ = &world_in;
  reference_table_ = arguments_in.reference_table_;
  if(arguments_in.query_table_ == NULL) {
    is_monochromatic_ = true;
    query_table_ = reference_table_;
  }
  else {
    is_monochromatic_ = false;
    query_table_ = arguments_in.query_table_;
  }

  // Declare the global constants.
  global_.Init(
    reference_table_, query_table_, reference_table_->n_entries(),
    arguments_in.bandwidth_, is_monochromatic_,
    arguments_in.relative_error_, arguments_in.probability_,
    arguments_in.kernel_, false);
  global_.set_effective_num_reference_points(
    world_in, reference_table_, query_table_);
}

template<typename DistributedTableType>
void DistributedAllknn<DistributedTableType>::set_bandwidth(
  double bandwidth_in) {
  global_.set_bandwidth(bandwidth_in);
}

template<typename DistributedTableType>
bool DistributedAllknn<DistributedTableType>::ConstructBoostVariableMap_(
  boost::mpi::communicator &world,
  const std::vector<std::string> &args,
  boost::program_options::variables_map *vm) {

  boost::program_options::options_description desc("Available options");
  desc.add_options()(
    "help", "Print this information."
  )(
    "references_in",
    boost::program_options::value<std::string>()->default_value(
      "random_dataset.csv"),
    "REQUIRED file containing reference data."
  )(
    "queries_in",
    boost::program_options::value<std::string>(),
    "OPTIONAL file containing query positions.  If omitted, allknn computes "
    "the leave-one-out neighbors at each reference point."
  )(
    "random_generate_n_attributes",
    boost::program_options::value<int>()->default_value(2),
    "Generate the datasets on the fly of the specified dimension."
  )(
    "random_generate_n_entries",
    boost::program_options::value<int>()->default_value(1000),
    "Generate the datasets on the fly of the specified number of points."
  )(
    "neighbors_out",
    boost::program_options::value<std::string>()->default_value(
      "neighbors_out.csv"),
    "OPTIONAL file to output the nearest neighbors."
  )(
    "leaf_size",
    boost::program_options::value<int>()->default_value(20),
    "Maximum number of points at a leaf of the tree."
  )(
    "use_memory_mapped_file",
    "Use memory mapped file for out-of-core computations."
  )(
    "memory_mapped_file_size",
    boost::program_options::value<unsigned int>(),
    "The size of the memory mapped file."
  )(
    "max_num_levels_to_serialize_in",
    boost::program_options::value<int>()->default_value(15),
    "The number of levels of subtrees to serialize at a given moment."
  )(
    "max_num_work_to_dequeue_per_stage_in",
    boost::program_options::value<int>()->default_value(5),
    "The number of work items to dequeue per process."
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

  // Validate the arguments. Only immediate dying is allowed here, the
  // parsing is done later.
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
  if(vm->count("references_in") == 0) {
    std::cerr << "Missing required --references_in.\n";
    exit(0);
  }
  if((*vm)["leaf_size"].as<int>() <= 0) {
    std::cerr << "The --leaf_size needs to be a positive integer.\n";
    exit(0);
  }
  if((*vm)["max_num_levels_to_serialize_in"].as<int>() <= 1) {
    std::cerr << "The --max_num_levels_to_serialize_in needs to be " <<
              "a positive integer greater than 1.\n";
    exit(0);
  }
  if((*vm)["max_num_work_to_dequeue_per_stage_in"].as<int>() <= 0) {
    std::cerr << "The --max_num_work_to_dequeue_per_stage_in needs to be " <<
              "a positive integer.\n";
    exit(0);
  }

  // Check whether the memory mapped file is being requested.
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
}

template<typename DistributedTableType>
void DistributedAllknn<DistributedTableType>::RandomGenerate(
  boost::mpi::communicator &world, const std::string &file_name,
  int num_dimensions, int num_points) {

  // Each process generates its own random data, dumps it to the file,
  // and read its own file back into its own distributed table.
  TableType random_dataset;
  random_dataset.Init(num_dimensions, num_points);
  for(int j = 0; j < num_points; j++) {
    core::table::DensePoint point;
    random_dataset.get(j, &point);
    for(int i = 0; i < num_dimensions; i++) {
      point[i] = core::math::Random(0.1, 1.0);
    }
  }
  printf("Process %d generated %d points...\n", world.rank(), num_points);
  random_dataset.Save(file_name);
}

template<typename DistributedTableType>
void DistributedAllknn<DistributedTableType>::ParseArguments(
  boost::mpi::communicator &world,
  const std::vector<std::string> &args,
  mlpack::distributed_allknn::DistributedAllknnArguments <
  DistributedTableType > *arguments_out) {

  // A L2 metric to index the table to use.
  arguments_out->metric_ = new core::metric_kernels::LMetric<2>();

  // Construct the Boost variable map.
  boost::program_options::variables_map vm;
  ConstructBoostVariableMap_(world, args, &vm);

  // Given the constructed boost variable map, parse each argument.

  // Parse the neighbors out file.
  arguments_out->neighbors_out_ = vm["neighbors_out"].as<std::string>();
  if(vm.count("random_generate_n_entries") > 0) {
    std::stringstream neighbors_out_sstr;
    neighbors_out_sstr << vm["neighbors_out"].as<std::string>() <<
                       world.rank();
    arguments_out->neighbors_out_ = neighbors_out_sstr.str();
  }

  // Parse the leaf size.
  arguments_out->leaf_size_ = vm["leaf_size"].as<int>();
  std::cout << "Using the leaf size of " << arguments_out->leaf_size_ << "\n";

  // Parse the reference set and index the tree.
  std::string reference_file_name = vm["references_in"].as<std::string>();
  if(vm.count("random_generate_n_entries") > 0) {
    std::stringstream reference_file_name_sstr;
    reference_file_name_sstr << vm["references_in"].as<std::string>() <<
                             world.rank();
    reference_file_name = reference_file_name_sstr.str();
    RandomGenerate(
      world, reference_file_name, vm["random_generate_n_attributes"].as<int>(),
      vm["random_generate_n_entries"].as<int>());
  }

  std::cout << "Reading in the reference set: " <<
            reference_file_name << "\n";
  arguments_out->reference_table_ =
    (core::table::global_m_file_) ?
    core::table::global_m_file_->Construct<DistributedTableType>() :
    new DistributedTableType();
  arguments_out->reference_table_->Init(
    reference_file_name, world);
  arguments_out->reference_table_->IndexData(
    *(arguments_out->metric_), world, arguments_out->leaf_size_,
    arguments_out->top_tree_sample_probability_);

  // Parse the query set and index the tree.
  if(vm.count("queries_in") > 0) {
    std::string query_file_name = vm["queries_in"].as<std::string>();
    if(vm.count("random_generate") > 0) {
      std::stringstream query_file_name_sstr;
      query_file_name_sstr << vm["queries_in"].as<std::string>() <<
                           world.rank();
      query_file_name = query_file_name_sstr.str();
      RandomGenerate(
        world, query_file_name, vm["random_generate_n_attributes"].as<int>(),
        vm["random_generate_n_entries"].as<int>());
    }
    std::cout << "Reading in the query set: " <<
              query_file_name << "\n";
    arguments_out->query_table_ =
      (core::table::global_m_file_) ?
      core::table::global_m_file_->Construct<DistributedTableType>() :
      new DistributedTableType();
    arguments_out->query_table_->Init(query_file_name, world);
    std::cout << "Finished reading in the query set.\n";
    std::cout << "Building the query tree.\n";
    arguments_out->query_table_->IndexData(
      *(arguments_out->metric_), world, arguments_out->leaf_size_,
      arguments_out->top_tree_sample_probability_);
    std::cout << "Finished building the query tree.\n";
  }

  // Parse the work parameters for the distributed engine.
  arguments_out->max_num_levels_to_serialize_ =
    vm["max_num_levels_to_serialize_in"].as<int>();
  arguments_out->max_num_work_to_dequeue_per_stage_ =
    vm["max_num_work_to_dequeue_per_stage_in"].as<int>();
  if(world.rank() == 0) {
    std::cout << "Serializing " << arguments_out->max_num_levels_to_serialize_
              << " levels of the tree at a time.\n";
    std::cout << "Dequeuing " <<
              arguments_out->max_num_work_to_dequeue_per_stage_ <<
              " items at a time from each process.\n";
  }
}

template<typename DistributedTableType>
void DistributedAllknn<DistributedTableType>::ParseArguments(
  int argc,
  char *argv[],
  boost::mpi::communicator &world,
  mlpack::distributed_allknn::DistributedAllknnArguments <
  DistributedTableType > *arguments_out) {

  // Convert C input to C++; skip executable name for Boost.
  std::vector<std::string> args(argv + 1, argv + argc);

  ParseArguments(world, args, arguments_out);
}
}
}

#endif

/** @file distributed_local_regression_dev.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_DISTRIBUTED_LOCAL_REGRESSION_DISTRIBUTED_LOCAL_REGRESSION_DEV_H
#define MLPACK_DISTRIBUTED_LOCAL_REGRESSION_DISTRIBUTED_LOCAL_REGRESSION_DEV_H

#include <omp.h>
#include "core/parallel/distributed_dualtree_dfs_dev.h"
#include "core/parallel/random_dataset_generator.h"
#include "core/metric_kernels/lmetric.h"
#include "core/table/memory_mapped_file.h"
#include "core/table/transform.h"
#include "mlpack/distributed_local_regression/distributed_local_regression.h"

namespace core {
namespace table {
extern core::table::MemoryMappedFile *global_m_file_;
}
}

namespace mlpack {
namespace distributed_local_regression {

template <
typename DistributedTableType, typename KernelType, typename MetricType >
DistributedTableType *DistributedLocalRegression <
DistributedTableType, KernelType, MetricType >::query_table() {
  return query_table_;
}

template <
typename DistributedTableType, typename KernelType, typename MetricType >
DistributedTableType *DistributedLocalRegression <
DistributedTableType, KernelType, MetricType >::reference_table() {
  return reference_table_;
}

template <
typename DistributedTableType, typename KernelType, typename MetricType >
typename DistributedLocalRegression <
DistributedTableType, KernelType, MetricType >::GlobalType &
DistributedLocalRegression <
DistributedTableType, KernelType, MetricType >::global() {
  return global_;
}

template <
typename DistributedTableType, typename KernelType, typename MetricType >
bool DistributedLocalRegression <
DistributedTableType, KernelType, MetricType >::is_monochromatic() const {
  return is_monochromatic_;
}

template <
typename DistributedTableType, typename KernelType, typename MetricType >
void DistributedLocalRegression <
DistributedTableType, KernelType, MetricType >::Compute(
  const mlpack::distributed_local_regression::
  DistributedLocalRegressionArguments <
  DistributedTableType, MetricType > &arguments_in,
  mlpack::local_regression::LocalRegressionResult *result_out) {

  // Barrier so that every process is here.
  world_->barrier();

  // Instantiate a dual-tree algorithm of the local regression.
  core::parallel::DistributedDualtreeDfs <
  mlpack::distributed_local_regression::DistributedLocalRegression <
  DistributedTableType, KernelType, MetricType > >
  distributed_dualtree_dfs;
  distributed_dualtree_dfs.Init(world_, *this);
  distributed_dualtree_dfs.set_work_params(
    arguments_in.leaf_size_,
    arguments_in.max_subtree_size_,
    arguments_in.max_num_work_to_dequeue_per_stage_);

  // Compute the result and do post-normalize.
  distributed_dualtree_dfs.Compute(arguments_in.metric_, result_out);
  result_out->PostProcess(global_, *(query_table_->local_table()));
}

template <
typename DistributedTableType, typename KernelType, typename MetricType >
void DistributedLocalRegression <
DistributedTableType, KernelType, MetricType >::Init(
  boost::mpi::communicator &world_in,
  mlpack::distributed_local_regression::DistributedLocalRegressionArguments <
  DistributedTableType, MetricType > &arguments_in) {

  world_ = &world_in;
  reference_table_ = arguments_in.reference_table_;
  if(arguments_in.query_table_ == NULL) {
    is_monochromatic_ = true;
    query_table_ = reference_table_;
    arguments_in.query_table_ = query_table_;
  }
  else {
    is_monochromatic_ = false;
    query_table_ = arguments_in.query_table_;
  }

  // Declare the global constants.
  arguments_in.do_postprocess_ = false;
  global_.Init(arguments_in);
  global_.set_effective_num_reference_points(
    world_in, reference_table_, query_table_);
}

template <
typename DistributedTableType, typename KernelType, typename MetricType >
void DistributedLocalRegression <
DistributedTableType, KernelType, MetricType >::set_bandwidth(
  double bandwidth_in) {
  global_.set_bandwidth(bandwidth_in);
}

bool DistributedLocalRegressionArgumentParser::ConstructBoostVariableMap(
  boost::mpi::communicator &world,
  const std::vector<std::string> &args,
  boost::program_options::variables_map *vm) {

  boost::program_options::options_description desc("Available options");
  desc.add_options()(
    "help", "Print this information."
  )(
    "absolute_error",
    boost::program_options::value<double>()->default_value(0.0),
    "Absolute error for the approximation of local regression per each "
    "query point."
  )(
    "bandwidth",
    boost::program_options::value<double>()->default_value(0.5),
    "OPTIONAL kernel bandwidth, if you set --bandwidth_selection flag, "
    "then the --bandwidth will be ignored."
  )(
    "kernel",
    boost::program_options::value<std::string>()->default_value("epan"),
    "Kernel function used by local regression.  One of:\n"
    "  epan, gaussian"
  )(
    "leaf_size",
    boost::program_options::value<int>()->default_value(40),
    "Maximum number of points at a leaf of the tree."
  )(
    "max_num_work_to_dequeue_per_stage_in",
    boost::program_options::value<int>()->default_value(30),
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
    "metric",
    boost::program_options::value<std::string>()->default_value("lmetric"),
    "Metric type used by local regression.  One of:\n"
    "  lmetric, weighted_lmetric"
  )(
    "metric_scales_in",
    boost::program_options::value<std::string>(),
    "The file containing the scaling factors for the weighted L2 metric."
  )(
    "num_threads_in",
    boost::program_options::value<int>()->default_value(1),
    "The number of threads to use for shared-memory parallelism."
  )(
    "order",
    boost::program_options::value<int>()->default_value(0),
    "The order of local polynomial to fit at each query point. One of: "
    "  0 or 1"
  )(
    "predictions_out",
    boost::program_options::value<std::string>()->default_value(
      "predictions_out.csv"),
    "OPTIONAL file to store the predicted regression values."
  )(
    "prescale",
    boost::program_options::value<std::string>()->default_value("none"),
    "OPTIONAL scaling option. One of:\n"
    "  none, hypercube, standardize"
  )(
    "probability",
    boost::program_options::value<double>()->default_value(1.0),
    "Probability guarantee for the approximation of local regression."
  )(
    "queries_in",
    boost::program_options::value<std::string>(),
    "OPTIONAL file containing query positions.  If omitted, local "
    "regression computes the leave-one-out density at each reference point."
  )(
    "random_generate",
    "If present, generate the dataset on the fly."
  )(
    "random_generate_n_attributes",
    boost::program_options::value<int>()->default_value(5),
    "Generate the datasets on the fly of the specified dimension."
  )(
    "random_generate_n_entries",
    boost::program_options::value<int>()->default_value(100000),
    "Generate the datasets on the fly of the specified number of points."
  )(
    "references_in",
    boost::program_options::value<std::string>()->default_value(
      "random_dataset.csv"),
    "REQUIRED file containing reference data."
  )(
    "reference_targets_in",
    boost::program_options::value<std::string>(),
    "The file containing the target values for the reference set."
  )(
    "relative_error",
    boost::program_options::value<double>()->default_value(0.1),
    "Relative error for the approximation of local regression."
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

  // Validate the arguments.
  if((*vm)["absolute_error"].as<double>() < 0.0) {
    std::cerr << "The --absolute_error requires a non-negative real "
              << "number.\n";
    exit(0);
  }
  if(vm->count("bandwidth") > 0 && (*vm)["bandwidth"].as<double>() <= 0) {
    std::cerr << "The --bandwidth requires a positive real number.\n";
    exit(0);
  }
  if(vm->count("bandwidth") == 0) {
    std::cerr << "Missing required --bandwidth.\n";
    exit(0);
  }
  if((*vm)["kernel"].as<std::string>() != "gaussian" &&
      (*vm)["kernel"].as<std::string>() != "epan") {
    std::cerr << "We support only epan or gaussian for the kernel.\n";
    exit(0);
  }
  if((*vm)["leaf_size"].as<int>() <= 0) {
    std::cerr << "The --leaf_size needs to be a positive integer.\n";
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
  if((*vm)["metric"].as<std::string>() == "weighted_lmetric" &&
      vm->count("metric_scales_in") == 0) {
    std::cerr << "The weighted L2 metric option requires " <<
              "--metric_scales_in option.\n";
    exit(0);
  }
  if((*vm)["num_threads_in"].as<int>() <= 0) {
    std::cerr << "The --num_threads_in needs to be a positive integer.\n";
    exit(0);
  }
  if((*vm)["order"].as<int>() < 0 || (*vm)["order"].as<int>() > 1) {
    std::cerr << "The --order requires either 0 (Nadaraya-Watson) or "
              "1 (local-linear).\n";
    exit(0);
  }
  if(vm->count("prescale") > 0) {
    if((*vm)["prescale"].as<std::string>() != "hypercube" &&
        (*vm)["prescale"].as<std::string>() != "standardize" &&
        (*vm)["prescale"].as<std::string>() != "none") {
      std::cerr << "The --prescale needs to be: none or hypercube or " <<
                "standardize.\n";
      exit(0);
    }
  }
  if((*vm)["probability"].as<double>() <= 0 ||
      (*vm)["probability"].as<double>() > 1) {
    std::cerr << "The --probability requires a real number $0 < p <= 1$.\n";
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
  if(vm->count("references_in") == 0) {
    std::cerr << "Missing required --references_in.\n";
    exit(0);
  }
  if(vm->count("reference_targets_in") == 0) {
    std::cerr << "Missing reqiured --reference_targets_in.\n";
    exit(0);
  }
  if((*vm)["relative_error"].as<double>() < 0) {
    std::cerr << "The --relative_error requires a real number $r >= 0$.\n";
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

template<typename TableType>
void DistributedLocalRegressionArgumentParser::RandomGenerate(
  boost::mpi::communicator &world,
  const std::string &file_name,
  const std::string *weight_file_name,
  int num_dimensions,
  int num_points,
  const std::string &prescale_option) {

  // Each process generates its own random data, dumps it to the file,
  // and read its own file back into its own distributed table.
  TableType random_dataset;
  core::parallel::RandomDatasetGenerator::Generate(
    num_dimensions, num_points, prescale_option, &random_dataset);
  random_dataset.Save(file_name, weight_file_name);
}

template<typename DistributedTableType, typename MetricType>
bool DistributedLocalRegressionArgumentParser::ParseArguments(
  boost::mpi::communicator &world,
  boost::program_options::variables_map &vm,
  mlpack::distributed_local_regression::DistributedLocalRegressionArguments <
  DistributedTableType, MetricType > *arguments_out) {

  // Parse the top tree sample probability.
  arguments_out->top_tree_sample_probability_ =
    vm["top_tree_sample_probability"].as<double>();
  if(world.rank() == 0) {
    std::cout << "Sampling the number of points owned by each MPI process with "
              "the probability of " <<
              arguments_out->top_tree_sample_probability_ << "\n";
  }

  // Parse the predictions out file.
  arguments_out->predictions_out_ = vm["predictions_out"].as<std::string>();
  if(vm.count("random_generate_n_entries") > 0) {
    std::stringstream predictions_out_sstr;
    predictions_out_sstr << vm["predictions_out"].as<std::string>() <<
                         world.rank();
    arguments_out->predictions_out_ = predictions_out_sstr.str();
  }

  // Parse the leaf size.
  arguments_out->leaf_size_ = vm["leaf_size"].as<int>();
  if(world.rank() == 0) {
    std::cout << "Using the leaf size of " << arguments_out->leaf_size_ << "\n";
  }

  // Parse the number of threads.
  arguments_out->num_threads_ = vm["num_threads_in"].as<int>();
  omp_set_num_threads(arguments_out->num_threads_);
  if(world.rank() == 0) {
    std::cout << "Using " << arguments_out->num_threads_ << " threads for " <<
              "shared memory parallelism.\n";
  }

  // Parse the reference set and index the tree.
  std::string reference_file_name = vm["references_in"].as<std::string>();
  std::string reference_targets_file_name =
    vm["reference_targets_in"].as<std::string>();
  if(vm.count("random_generate") > 0) {
    std::stringstream reference_file_name_sstr;
    std::stringstream reference_targets_file_name_sstr;
    reference_file_name_sstr << vm["references_in"].as<std::string>() <<
                             world.rank();
    reference_file_name = reference_file_name_sstr.str();
    reference_targets_file_name_sstr <<
                                     vm["reference_targets_in"].as<std::string>() << world.rank();
    reference_targets_file_name = reference_targets_file_name_sstr.str();

    RandomGenerate<typename DistributedTableType::TableType>(
      world, reference_file_name,
      &reference_targets_file_name,
      vm["random_generate_n_attributes"].as<int>(),
      vm["random_generate_n_entries"].as<int>(),
      vm["prescale"].as<std::string>());
  }

  std::cout << "Reading in the reference set: " <<
            reference_file_name << "\n";
  arguments_out->reference_table_ =
    (core::table::global_m_file_) ?
    core::table::global_m_file_->Construct<DistributedTableType>() :
    new DistributedTableType();
  arguments_out->reference_table_->Init(
    reference_file_name, world, &reference_targets_file_name);
  arguments_out->reference_table_->IndexData(
    arguments_out->metric_, world, arguments_out->leaf_size_,
    arguments_out->top_tree_sample_probability_);

  // Parse the query set and index the tree.
  if(vm.count("queries_in") > 0) {
    std::string query_file_name = vm["queries_in"].as<std::string>();
    if(vm.count("random_generate") > 0) {
      std::stringstream query_file_name_sstr;
      query_file_name_sstr << vm["queries_in"].as<std::string>() <<
                           world.rank();
      query_file_name = query_file_name_sstr.str();
      RandomGenerate<typename DistributedTableType::TableType>(
        world, query_file_name,
        (const std::string *) NULL,
        vm["random_generate_n_attributes"].as<int>(),
        vm["random_generate_n_entries"].as<int>(),
        vm["prescale"].as<std::string>());
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
      arguments_out->metric_, world, arguments_out->leaf_size_,
      arguments_out->top_tree_sample_probability_);
    std::cout << "Finished building the query tree.\n";
  }

  // Parse the bandwidth.
  arguments_out->bandwidth_ = vm["bandwidth"].as<double>();
  if(world.rank() == 0) {
    std::cout << "Bandwidth of " << arguments_out->bandwidth_ << "\n";
  }

  // Parse the relative error and the absolute error.
  arguments_out->absolute_error_ = vm["absolute_error"].as<double>();
  arguments_out->relative_error_ = vm["relative_error"].as<double>();
  if(world.rank() == 0) {
    std::cout << "For each query point $q \\in \\mathcal{Q}$, " <<
              "we will guarantee: " <<
              "$| \\widetilde{G}(q) - G(q) | \\leq "
              << arguments_out->relative_error_ <<
              " \\cdot G(q) + " << arguments_out->absolute_error_ <<
              " | \\mathcal{R} |$ \n";
  }

  // Parse the order and determine the problem dimension.
  arguments_out->order_ = vm["order"].as<int>();
  arguments_out->problem_dimension_ =
    (arguments_out->order_ == 0) ?
    1 : arguments_out->reference_table_->n_attributes() + 1;
  if(world.rank() == 0) {
    std::cout << "The order of local polynomial approximation: " <<
              arguments_out->order_ << "\n";
  }

  // Parse the probability.
  arguments_out->probability_ = vm["probability"].as<double>();
  if(world.rank() == 0) {
    std::cout << "Probability of " << arguments_out->probability_ << "\n";
  }

  // Parse the kernel type.
  arguments_out->kernel_ = vm["kernel"].as< std::string >();
  if(world.rank() == 0) {
    std::cout << "Using the kernel: " << arguments_out->kernel_ << "\n";
  }

  // Parse the work parameters for the distributed engine.
  arguments_out->max_subtree_size_ =
    std::min(
      vm["max_subtree_size_in"].as<int>(),
      arguments_out->reference_table_->n_entries() /
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

bool DistributedLocalRegressionArgumentParser::ConstructBoostVariableMap(
  boost::mpi::communicator &world,
  int argc,
  char *argv[],
  boost::program_options::variables_map *vm) {

  // Convert C input to C++; skip executable name for Boost.
  std::vector<std::string> args(argv + 1, argv + argc);

  return ConstructBoostVariableMap(world, args, vm);
}
}
}

#endif

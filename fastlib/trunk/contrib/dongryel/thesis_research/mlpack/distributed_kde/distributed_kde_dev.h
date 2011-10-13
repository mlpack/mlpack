/** @file distributed_kde_dev.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_DISTRIBUTED_KDE_DISTRIBUTED_KDE_DEV_H
#define MLPACK_DISTRIBUTED_KDE_DISTRIBUTED_KDE_DEV_H

#include <omp.h>
#include "core/parallel/distributed_dualtree_dfs_dev.h"
#include "core/parallel/random_dataset_generator.h"
#include "core/metric_kernels/lmetric.h"
#include "core/table/memory_mapped_file.h"
#include "core/table/transform.h"
#include "mlpack/distributed_kde/distributed_kde.h"

namespace core {
namespace table {
extern core::table::MemoryMappedFile *global_m_file_;
}
}

namespace mlpack {
namespace distributed_kde {

template<typename DistributedTableType, typename KernelAuxType>
DistributedTableType *DistributedKde <
DistributedTableType, KernelAuxType >::query_table() {
  return query_table_;
}

template<typename DistributedTableType, typename KernelAuxType>
DistributedTableType *DistributedKde <
DistributedTableType, KernelAuxType >::reference_table() {
  return reference_table_;
}

template<typename DistributedTableType, typename KernelAuxType>
typename DistributedKde<DistributedTableType, KernelAuxType>::GlobalType &
DistributedKde<DistributedTableType, KernelAuxType>::global() {
  return global_;
}

template<typename DistributedTableType, typename KernelAuxType>
bool DistributedKde <
DistributedTableType, KernelAuxType >::is_monochromatic() const {
  return is_monochromatic_;
}

template<typename DistributedTableType, typename KernelAuxType>
void DistributedKde<DistributedTableType, KernelAuxType>::Compute(
  const mlpack::distributed_kde::DistributedKdeArguments <
  DistributedTableType > &arguments_in,
  ResultType *result_out) {

  // Barrier so that every process is here.
  world_->barrier();

  // Instantiate a dual-tree algorithm of the KDE.
  core::parallel::DistributedDualtreeDfs <
  mlpack::distributed_kde::DistributedKde <
  DistributedTableType, KernelAuxType > >
  distributed_dualtree_dfs;
  distributed_dualtree_dfs.Init(world_, *this);
  distributed_dualtree_dfs.set_work_params(
    arguments_in.leaf_size_,
    arguments_in.max_subtree_size_,
    arguments_in.do_load_balancing_,
    arguments_in.max_num_work_to_dequeue_per_stage_);

  // If weak-scaling measuring is requested, then turn it on.
  if(arguments_in.measure_weak_scaling_) {
    distributed_dualtree_dfs.enable_weak_scaling_measuring_mode(
      arguments_in.weak_scaling_factor_);
  }

  // Compute the result and do post-normalize.
  distributed_dualtree_dfs.Compute(* arguments_in.metric_, result_out);
  result_out->Normalize(global_);
}

template<typename DistributedTableType, typename KernelAuxType>
void DistributedKde<DistributedTableType, KernelAuxType>::Init(
  boost::mpi::communicator &world_in,
  mlpack::distributed_kde::DistributedKdeArguments <
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
    (KernelAuxType *) NULL, arguments_in.bandwidth_,
    (typename GlobalType::MeanVariancePairListType *) NULL, is_monochromatic_,
    arguments_in.relative_error_, arguments_in.absolute_error_,
    arguments_in.probability_, false);
  global_.set_effective_num_reference_points(
    world_in, reference_table_, query_table_);
}

template<typename DistributedTableType, typename KernelAuxType>
void DistributedKde<DistributedTableType, KernelAuxType>::set_bandwidth(
  double bandwidth_in) {
  global_.set_bandwidth(bandwidth_in);
}

bool DistributedKdeArgumentParser::ConstructBoostVariableMap(
  boost::mpi::communicator &world,
  const std::vector<std::string> &args,
  boost::program_options::variables_map *vm) {

  boost::program_options::options_description desc("Available options");
  desc.add_options()(
    "help", "Print this information."
  )(
    "absolute_error",
    boost::program_options::value<double>()->default_value(0.0),
    "Absolute error for the approximation of KDE per each query point."
  )(
    "bandwidth",
    boost::program_options::value<double>()->default_value(0.5),
    "OPTIONAL kernel bandwidth, if you set --bandwidth_selection flag, "
    "then the --bandwidth will be ignored."
  )(
    "densities_out",
    boost::program_options::value<std::string>()->default_value(
      "densities_out.csv"),
    "OPTIONAL file to store computed densities."
  )(
    "do_load_balancing",
    "If present, do load balancing."
  )(
    "kernel",
    boost::program_options::value<std::string>()->default_value("gaussian"),
    "Kernel function used by KDE.  One of:\n"
    "  epan, gaussian"
  )(
    "leaf_size",
    boost::program_options::value<int>()->default_value(4),
    "Maximum number of points at a leaf of the tree."
  )(
    "max_num_work_to_dequeue_per_stage_in",
    boost::program_options::value<int>()->default_value(5),
    "The number of work items to dequeue per process."
  )(
    "max_subtree_size_in",
    boost::program_options::value<int>()->default_value(20),
    "The maximum size of the subtree to serialize at a given moment."
  )(
    "measure_weak_scaling",
    "If present, approximation tolerance is ignored and will enable the "
    "weak-scaling measurement mode."
  )(
    "memory_mapped_file_size",
    boost::program_options::value<unsigned int>(),
    "The size of the memory mapped file."
  )(
    "num_threads_in",
    boost::program_options::value<int>()->default_value(1),
    "The number of threads to use for shared-memory parallelism."
  )(
    "probability",
    boost::program_options::value<double>()->default_value(1.0),
    "Probability guarantee for the approximation of KDE."
  )(
    "prescale",
    boost::program_options::value<std::string>()->default_value("none"),
    "OPTIONAL scaling option. One of:\n"
    "  none, hypercube, standardize"
  )(
    "queries_in",
    boost::program_options::value<std::string>(),
    "OPTIONAL file containing query positions.  If omitted, KDE computes "
    "the leave-one-out density at each reference point."
  )(
    "random_generate",
    "If present, generate the datasets on the fly."
  )(
    "random_generate_n_attributes",
    boost::program_options::value<int>()->default_value(5),
    "Generate the datasets on the fly of the specified dimension."
  )(
    "random_generate_n_entries",
    boost::program_options::value<int>()->default_value(1000),
    "Generate the datasets on the fly of the specified number of points."
  )(
    "random_seed_in",
    boost::program_options::value<unsigned long int>(),
    "Random seed to start for each MPI process."
  )(
    "references_in",
    boost::program_options::value<std::string>()->default_value(
      "random_dataset.csv"),
    "REQUIRED file containing reference data."
  )(
    "relative_error",
    boost::program_options::value<double>()->default_value(0.1),
    "Relative error for the approximation of KDE."
  )(
    "series_expansion_type",
    boost::program_options::value<std::string>()->default_value("multivariate"),
    "Series expansion type used to compress the kernel interaction. One of:\n"
    "  hypercube, multivariate"
  )(
    "top_tree_sample_probability",
    boost::program_options::value<double>()->default_value(0.3),
    "The portion of points sampled on each MPI process for building the "
    "top tree."
  )(
    "tree",
    boost::program_options::value<std::string>()->default_value("kdtree"),
    "The tree type used in the computation. One of:\n"
    "  kdtree, metrictree"
  )(
    "use_memory_mapped_file",
    "Use memory mapped file for out-of-core computations."
  )(
    "weak_scaling_factor",
    boost::program_options::value<double>()->default_value(0.05),
    "The percentage of reference points to consider in weak scaling measuring."
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
    std::cerr << desc << "\n";
    return true;
  }

  // Validate the arguments.
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
  if((*vm)["num_threads_in"].as<int>() <= 0) {
    std::cerr << "The --num_threads_in needs to be a positive integer.\n";
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
  if((*vm)["relative_error"].as<double>() < 0) {
    std::cerr << "The --relative_error requires a real number $r >= 0$.\n";
    exit(0);
  }
  if((*vm)["series_expansion_type"].as<std::string>() != "hypercube" &&
      (*vm)["series_expansion_type"].as<std::string>() != "multivariate") {
    std::cerr << "We support only hypercube or multivariate for the "
              "series expansion type.\n";
    exit(0);
  }
  if((*vm)["tree"].as<std::string>() != "kdtree" &&
      (*vm)["tree"].as<std::string>() != "metrictree") {
    std::cerr <<
              "We support only the kdtree or metrictree for the tree type.\n";
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
bool DistributedKdeArgumentParser::ParseArguments(
  boost::mpi::communicator &world,
  boost::program_options::variables_map &vm,
  mlpack::distributed_kde::DistributedKdeArguments <
  DistributedTableType > *arguments_out) {

  // Define the table type.
  typedef typename DistributedTableType::TableType TableType;

  // A L2 metric to index the table to use.
  arguments_out->metric_ = new core::metric_kernels::LMetric<2>();

  // Parse the load balancing option.
  arguments_out->do_load_balancing_ = (true || vm.count("do_load_balancing") > 0);

  // Parse the weak scaling measuring option.
  arguments_out->measure_weak_scaling_ =
    (vm.count("measure_weak_scaling") > 0);
  if(arguments_out->measure_weak_scaling_) {
    arguments_out->weak_scaling_factor_ = vm[ "weak_scaling_factor" ].as<double>();
  }

  // Parse the top tree sample probability.
  arguments_out->top_tree_sample_probability_ =
    vm["top_tree_sample_probability"].as<double>();
  if(world.rank() == 0) {
    std::cerr << "Sampling the number of points owned by each MPI process with "
              "the probability of " <<
              arguments_out->top_tree_sample_probability_ << "\n";
  }

  // Parse the densities out file.
  arguments_out->densities_out_ = vm["densities_out"].as<std::string>();
  if(vm.count("random_generate_n_entries") > 0) {
    std::stringstream densities_out_sstr;
    densities_out_sstr << vm["densities_out"].as<std::string>() <<
                       world.rank();
    arguments_out->densities_out_ = densities_out_sstr.str();
  }

  // Parse the leaf size.
  arguments_out->leaf_size_ = vm["leaf_size"].as<int>();
  if(world.rank() == 0) {
    std::cerr << "Using the leaf size of " << arguments_out->leaf_size_ << "\n";
  }

  // Parse the number of threads.
  arguments_out->num_threads_ = vm["num_threads_in"].as<int>();
  omp_set_num_threads(arguments_out->num_threads_);
  std::cerr << "  Process " << world.rank() << " is using " <<
            arguments_out->num_threads_ << " threads for " <<
            "shared memory parallelism.\n";

  // Parse the random seed if it is available and set it.
  if(vm.count("random_seed_in") > 0) {
    unsigned long int seed = vm["random_seed_in"].as<unsigned long int>();
    core::math::global_random_number_state_.set_seed(seed + world.rank());
  }

  // Parse the reference set and index the tree.
  std::string reference_file_name = vm["references_in"].as<std::string>();
  arguments_out->reference_table_ =
    (core::table::global_m_file_) ?
    core::table::global_m_file_->Construct<DistributedTableType>() :
    new DistributedTableType();
  if(true || vm.count("random_generate") > 0) {
    std::stringstream reference_file_name_sstr;
    reference_file_name_sstr << vm["references_in"].as<std::string>() <<
                             world.rank();
    reference_file_name = reference_file_name_sstr.str();
    TableType *random_reference_dataset =
      (core::table::global_m_file_) ?
      core::table::global_m_file_->Construct<TableType>() : new TableType();
    core::parallel::RandomDatasetGenerator::Generate(
      vm["random_generate_n_attributes"].as<int>(),
      vm["random_generate_n_entries"].as<int>(), world.rank(),
      vm["prescale"].as<std::string>(), false, random_reference_dataset);
    arguments_out->reference_table_->Init(random_reference_dataset, world);
  }
  else {
    std::cerr << "Reading in the reference set: " <<
              reference_file_name << "\n";
    arguments_out->reference_table_->Init(
      reference_file_name, world);
  }
  arguments_out->reference_table_->IndexData(
    *(arguments_out->metric_), world, arguments_out->leaf_size_,
    arguments_out->top_tree_sample_probability_, 0);

  // Parse the query set and index the tree.
  if(vm.count("queries_in") > 0) {
    std::string query_file_name = vm["queries_in"].as<std::string>();
    arguments_out->query_table_ =
      (core::table::global_m_file_) ?
      core::table::global_m_file_->Construct<DistributedTableType>() :
      new DistributedTableType();
    if(vm.count("random_generate") > 0) {
      std::stringstream query_file_name_sstr;
      query_file_name_sstr << vm["queries_in"].as<std::string>() <<
                           world.rank();
      query_file_name = query_file_name_sstr.str();
      TableType *random_query_dataset =
        (core::table::global_m_file_) ?
        core::table::global_m_file_->Construct<TableType>() : new TableType();
      core::parallel::RandomDatasetGenerator::Generate(
        vm["random_generate_n_attributes"].as<int>(),
        vm["random_generate_n_entries"].as<int>(), world.rank(),
        vm["prescale"].as<std::string>(), false, random_query_dataset);
      arguments_out->query_table_->Init(random_query_dataset, world);
    }
    else {
      std::cerr << "Reading in the query set: " <<
                query_file_name << "\n";
      arguments_out->query_table_->Init(query_file_name, world);
    }
    std::cerr << "Building the query tree.\n";
    arguments_out->query_table_->IndexData(
      *(arguments_out->metric_), world, arguments_out->leaf_size_,
      arguments_out->top_tree_sample_probability_, 1);
    std::cerr << "Finished building the query tree.\n";
  }

  // Parse the bandwidth.
  arguments_out->bandwidth_ = vm["bandwidth"].as<double>();
  if(world.rank() == 0) {
    std::cerr << "Bandwidth of " << arguments_out->bandwidth_ << "\n";
  }

  // Parse the relative error and the absolute error.
  arguments_out->absolute_error_ = vm["absolute_error"].as<double>();
  arguments_out->relative_error_ = vm["relative_error"].as<double>();
  if(world.rank() == 0) {
    std::cerr << "For each query point $q \\in \\mathcal{Q}$, " <<
              "we will guarantee: " <<
              "$| \\widetilde{G}(q) - G(q) | \\leq "
              << arguments_out->relative_error_ <<
              " \\cdot G(q) + " << arguments_out->absolute_error_ <<
              " | \\mathcal{R} |$ \n";
  }

  // Parse the probability.
  arguments_out->probability_ = vm["probability"].as<double>();
  if(world.rank() == 0) {
    std::cerr << "Probability of " << arguments_out->probability_ << "\n";
  }

  // Parse the kernel type.
  arguments_out->kernel_ = vm["kernel"].as< std::string >();
  if(world.rank() == 0) {
    std::cerr << "Using the kernel: " << arguments_out->kernel_ << "\n";
  }

  // Parse the series expansion type.
  arguments_out->series_expansion_type_ =
    vm["series_expansion_type"].as<std::string>();
  if(world.rank() == 0) {
    std::cerr << "Using the series expansion type: " <<
              arguments_out->series_expansion_type_ << "\n";
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
    std::cerr << "Serializing " << arguments_out->max_subtree_size_
              << " points of the tree at a time.\n";
    std::cerr << "Dequeuing " <<
              arguments_out->max_num_work_to_dequeue_per_stage_ <<
              " items at a time from each process.\n";
  }

  return false;
}

bool DistributedKdeArgumentParser::ConstructBoostVariableMap(
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

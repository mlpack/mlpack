/** @file distributed_kde_dev.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_DISTRIBUTED_KDE_DISTRIBUTED_KDE_DEV_H
#define MLPACK_DISTRIBUTED_KDE_DISTRIBUTED_KDE_DEV_H

#include "core/gnp/distributed_dualtree_dfs_dev.h"
#include "core/metric_kernels/lmetric.h"
#include "core/table/memory_mapped_file.h"
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
  mlpack::kde::KdeResult< std::vector<double> > *result_out) {

  // Barrier so that every process is here.
  world_->barrier();

  boost::mpi::timer timer;

  // Instantiate a dual-tree algorithm of the KDE.
  core::gnp::DistributedDualtreeDfs <
  mlpack::distributed_kde::DistributedKde <
  DistributedTableType, KernelAuxType > >
  distributed_dualtree_dfs;
  distributed_dualtree_dfs.Init(world_, *this);
  distributed_dualtree_dfs.set_work_params(
    arguments_in.leaf_size_,
    arguments_in.max_num_levels_to_serialize_,
    arguments_in.max_num_work_to_dequeue_per_stage_);

  // Compute the result and do post-normalize.
  distributed_dualtree_dfs.Compute(* arguments_in.metric_, result_out);
  result_out->Normalize(global_);

  if(world_->rank() == 0) {
    printf("Spent %g seconds in computation.\n", timer.elapsed());
  }
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
    "references_in",
    boost::program_options::value<std::string>()->default_value(
      "random_dataset.csv"),
    "REQUIRED file containing reference data."
  )(
    "queries_in",
    boost::program_options::value<std::string>(),
    "OPTIONAL file containing query positions.  If omitted, KDE computes "
    "the leave-one-out density at each reference point."
  )(
    "random_generate_n_attributes",
    boost::program_options::value<int>()->default_value(5),
    "Generate the datasets on the fly of the specified dimension."
  )(
    "random_generate_n_entries",
    boost::program_options::value<int>()->default_value(100000),
    "Generate the datasets on the fly of the specified number of points."
  )(
    "densities_out",
    boost::program_options::value<std::string>()->default_value(
      "densities_out.csv"),
    "OPTIONAL file to store computed densities."
  )(
    "kernel",
    boost::program_options::value<std::string>()->default_value("gaussian"),
    "Kernel function used by KDE.  One of:\n"
    "  epan, gaussian"
  )(
    "series_expansion_type",
    boost::program_options::value<std::string>()->default_value("hypercube"),
    "Series expansion type used to compress the kernel interaction. One of:\n"
    "  hypercube, multivariate"
  )(
    "bandwidth",
    boost::program_options::value<double>()->default_value(0.5),
    "OPTIONAL kernel bandwidth, if you set --bandwidth_selection flag, "
    "then the --bandwidth will be ignored."
  )
  (
    "probability",
    boost::program_options::value<double>()->default_value(1.0),
    "Probability guarantee for the approximation of KDE."
  )(
    "absolute_error",
    boost::program_options::value<double>()->default_value(0.0),
    "Absolute error for the approximation of KDE per each query point."
  )(
    "relative_error",
    boost::program_options::value<double>()->default_value(0.1),
    "Relative error for the approximation of KDE."
  )(
    "leaf_size",
    boost::program_options::value<int>()->default_value(40),
    "Maximum number of points at a leaf of the tree."
  )(
    "top_tree_sample_probability",
    boost::program_options::value<double>()->default_value(0.3),
    "The portion of points sampled on each MPI process for building the "
    "top tree."
  )(
    "use_memory_mapped_file",
    "Use memory mapped file for out-of-core computations."
  )(
    "memory_mapped_file_size",
    boost::program_options::value<unsigned int>(),
    "The size of the memory mapped file."
  )(
    "max_num_levels_to_serialize_in",
    boost::program_options::value<int>()->default_value(20),
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
  if((*vm)["kernel"].as<std::string>() != "gaussian" &&
      (*vm)["kernel"].as<std::string>() != "epan") {
    std::cerr << "We support only epan or gaussian for the kernel.\n";
    exit(0);
  }
  if((*vm)["series_expansion_type"].as<std::string>() != "hypercube" &&
      (*vm)["series_expansion_type"].as<std::string>() != "multivariate") {
    std::cerr << "We support only hypercube or multivariate for the "
              "series expansion type.\n";
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
  if((*vm)["probability"].as<double>() <= 0 ||
      (*vm)["probability"].as<double>() > 1) {
    std::cerr << "The --probability requires a real number $0 < p <= 1$.\n";
    exit(0);
  }
  if((*vm)["relative_error"].as<double>() < 0) {
    std::cerr << "The --relative_error requires a real number $r >= 0$.\n";
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

template<typename TableType>
void DistributedKdeArgumentParser::RandomGenerate(
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
  printf("Process %d generated %d points in %d dimensionality...\n",
         world.rank(), num_points, num_dimensions);
  random_dataset.Save(file_name);
}

template<typename DistributedTableType>
bool DistributedKdeArgumentParser::ParseArguments(
  boost::mpi::communicator &world,
  boost::program_options::variables_map &vm,
  mlpack::distributed_kde::DistributedKdeArguments <
  DistributedTableType > *arguments_out) {

  // A L2 metric to index the table to use.
  arguments_out->metric_ = new core::metric_kernels::LMetric<2>();

  // Parse the top tree sample probability.
  arguments_out->top_tree_sample_probability_ =
    vm["top_tree_sample_probability"].as<double>();
  if(world.rank() == 0) {
    std::cout << "Sampling the number of points owned by each MPI process with "
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
  std::cout << "Using the leaf size of " << arguments_out->leaf_size_ << "\n";

  // Parse the reference set and index the tree.
  std::string reference_file_name = vm["references_in"].as<std::string>();
  if(vm.count("random_generate_n_entries") > 0) {
    std::stringstream reference_file_name_sstr;
    reference_file_name_sstr << vm["references_in"].as<std::string>() <<
                             world.rank();
    reference_file_name = reference_file_name_sstr.str();
    RandomGenerate<typename DistributedTableType::TableType>(
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
      RandomGenerate<typename DistributedTableType::TableType>(
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

  // Parse the series expansion type.
  arguments_out->series_expansion_type_ =
    vm["series_expansion_type"].as<std::string>();
  if(world.rank() == 0) {
    std::cout << "Using the series expansion type: " <<
              arguments_out->series_expansion_type_ << "\n";
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

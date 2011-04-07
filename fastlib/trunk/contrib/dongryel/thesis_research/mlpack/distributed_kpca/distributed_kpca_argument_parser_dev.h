/** @file distributed_kpca_argument_parser_dev.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_DISTRIBUTED_KPCA_DISTRIBUTED_KPCA_ARGUMENT_PARSER_DEV_H
#define MLPACK_DISTRIBUTED_KPCA_DISTRIBUTED_KPCA_ARGUMENT_PARSER_DEV_H

#include "core/csv_parser/dataset_reader.h"
#include "mlpack/distributed_kpca/distributed_kpca.h"

namespace mlpack {
namespace distributed_kpca {

bool DistributedKpcaArgumentParser::ConstructBoostVariableMap(
  boost::mpi::communicator &world,
  const std::vector<std::string> &args,
  boost::program_options::variables_map *vm) {

  boost::program_options::options_description desc("Available options");
  desc.add_options()(
    "help", "Print this information."
  )(
    "do_centering", "Apply centering to KPCA if present."
  )(
    "do_naive", "Do naive computations along with the fast for debugging "
    "purposes. Only applies to the case when there is only one MPI process "
    "present."
  )(
    "mode",
    boost::program_options::value<std::string>()->default_value("kde"),
    "OPTIONAL The algorithm mode. One of:"
    "  kde, kpca"
  )(
    "num_kpca_components_in",
    boost::program_options::value<int>()->default_value(3),
    "OPTIONAL The number of KPCA components to output."
  )(
    "kpca_components_out",
    boost::program_options::value<std::string>()->default_value(
      "kpca_components.csv"),
    "OPTIONAL output file for KPCA components."
  )(
    "kpca_projections_out",
    boost::program_options::value<std::string>()->default_value(
      "kpca_projections.csv"),
    "OPTIONAL output file for KPCA projections."
  )(
    "max_num_iterations_in",
    boost::program_options::value<int>()->default_value(30),
    "OPTIONAL The number of maximum sampling rounds."
  )(
    "references_in",
    boost::program_options::value<std::string>()->default_value(
      "random_dataset.csv"),
    "REQUIRED file containing reference data."
  )(
    "queries_in",
    boost::program_options::value<std::string>(),
    "OPTIONAL file containing query positions."
  )(
    "random_generate_n_attributes",
    boost::program_options::value<int>(),
    "OPTIONAL Generate the datasets on the fly of the specified dimension."
  )(
    "random_generate_n_entries",
    boost::program_options::value<int>(),
    "OPTIONAL Generate the datasets on the fly of the specified number "
    "of points."
  )(
    "kernel",
    boost::program_options::value<std::string>()->default_value("gaussian"),
    "Kernel function used by KPCA.  One of:\n"
    "  gaussian"
  )(
    "bandwidth",
    boost::program_options::value<double>()->default_value(0.5),
    "OPTIONAL kernel bandwidth, if you set --bandwidth_selection flag, "
    "then the --bandwidth will be ignored."
  )(
    "num_threads_in",
    boost::program_options::value<int>()->default_value(4),
    "OPTIONAL The level of shared-memory parallelism."
  )(
    "probability",
    boost::program_options::value<double>()->default_value(0.9),
    "Probability guarantee for the approximation of KPCA."
  )(
    "absolute_error",
    boost::program_options::value<double>()->default_value(1e-6),
    "Absolute error for the approximation of KPCA per each query point."
  )(
    "relative_error",
    boost::program_options::value<double>()->default_value(0.1),
    "Relative error for the approximation of KPCA."
  )(
    "use_memory_mapped_file",
    "Use memory mapped file for out-of-core computations."
  )(
    "memory_mapped_file_size",
    boost::program_options::value<unsigned int>(),
    "The size of the memory mapped file."
  )(
    "prescale",
    boost::program_options::value<std::string>()->default_value("none"),
    "OPTIONAL scaling option. One of:\n"
    "  none, hypercube, standardize"
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

  // Validate the arguments. Only immediate termination is allowed
  // here, the parsing is done later.
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
  if(vm->count("mode") > 0) {
    if((*vm)["mode"].as<std::string>() != "kde" &&
        (*vm)["mode"].as<std::string>() != "kpca") {
      std::cerr << "The mode supports either kde or kpca.\n";
      exit(0);
    }
  }
  if(vm->count("references_in") == 0) {
    std::cerr << "Missing required --references_in.\n";
    exit(0);
  }
  if((*vm)["kernel"].as<std::string>() != "gaussian") {
    std::cerr << "We support only gaussian for the kernel.\n";
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
  if((*vm)["num_kpca_components_in"].as<int>() <= 0) {
    std::cerr << "The --num_kpca_components_in requires an integer > 0.\n";
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

    // Delete the temporary files and put a barrier.
    std::stringstream temporary_file_name;
    temporary_file_name << "tmp_file" << world.rank();
    remove(temporary_file_name.str().c_str());
    world.barrier();

    // Initialize the memory allocator.
    core::table::global_m_file_ = new core::table::MemoryMappedFile();
    core::table::global_m_file_->Init(
      std::string("tmp_file"), world.rank(), world.rank(), 100000000);
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

  return false;
}

template<typename TableType>
void DistributedKpcaArgumentParser::RandomGenerate(
  boost::mpi::communicator &world, const std::string &file_name,
  int num_dimensions, int num_points, const std::string &prescale_option) {

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

  // Scale the dataset.
  if(prescale_option == "hypercube") {
    core::table::UnitHypercube::Transform(&random_dataset);
  }
  else if(prescale_option == "standardize") {
    core::table::Standardize::Transform(&random_dataset);
  }
  if(world.rank() == 0) {
    std::cout << "Scaled the dataset with the option: " <<
              prescale_option << "\n";
  }

  random_dataset.Save(file_name);
}

template<typename DistributedTableType>
bool DistributedKpcaArgumentParser::ParseArguments(
  boost::mpi::communicator &world,
  boost::program_options::variables_map &vm,
  mlpack::distributed_kpca::DistributedKpcaArguments <
  DistributedTableType > *arguments_out) {

  // Parse the reference set and index the tree.
  std::string reference_file_name = vm["references_in"].as<std::string>();
  if(vm.count("random_generate_n_entries") > 0) {
    std::stringstream reference_file_name_sstr;
    reference_file_name_sstr << vm["references_in"].as<std::string>() <<
                             world.rank();
    reference_file_name = reference_file_name_sstr.str();
    RandomGenerate<typename DistributedTableType::TableType>(
      world, reference_file_name, vm["random_generate_n_attributes"].as<int>(),
      vm["random_generate_n_entries"].as<int>(),
      vm["prescale"].as<std::string>());
  }

  // This is a hack to make sure that the same dataset is split across
  // multiple processes when there are more than one MPI process.
  else if(world.size() > 0) {

    // Only the master splits the file.
    if(world.rank() == 0) {
      std::cout << "Splitting the file into parts...\n";
      core::DatasetReader::SplitFile <
      typename DistributedTableType::TableType > (
        reference_file_name, world.size());
    }
    world.barrier();
    std::stringstream reference_file_name_sstr;
    reference_file_name_sstr << vm["references_in"].as<std::string>() <<
                             world.rank();
    reference_file_name = reference_file_name_sstr.str();
  }

  std::cout << "Reading in the reference set: " <<
            reference_file_name << "\n";
  arguments_out->reference_table_ =
    (core::table::global_m_file_) ?
    core::table::global_m_file_->Construct<DistributedTableType>() :
    new DistributedTableType();
  arguments_out->reference_table_->Init(
    reference_file_name, world);

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
        vm["random_generate_n_entries"].as<int>(),
        vm["prescale"].as<std::string>());
    }

    // A hack to split the query file into multiple files across each
    // MPI process.
    else {

      // Only the master splits the file.
      if(world.rank() == 0) {
        std::cout << "Splitting the file into parts...\n";
        core::DatasetReader::SplitFile <
        typename DistributedTableType::TableType > (
          query_file_name, world.size());
      }
      world.barrier();
      std::stringstream query_file_name_sstr;
      query_file_name_sstr << query_file_name << world.rank();
      query_file_name = query_file_name_sstr.str();
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

  // Parse the KPCA component output file.
  arguments_out->kpca_components_out_ =
    vm["kpca_components_out"].as<std::string>();
  if(vm.count("random_generate_n_entries") > 0 || world.size() > 1) {
    std::stringstream kpca_components_out_sstr;
    kpca_components_out_sstr << vm["kpca_components_out"].as<std::string>() <<
                             world.rank();
    arguments_out->kpca_components_out_ = kpca_components_out_sstr.str();
  }

  // Parse the KPCA projection output file.
  arguments_out->kpca_projections_out_ =
    vm["kpca_projections_out"].as<std::string>();
  if(vm.count("random_generate_n_entries") > 0 || world.size() > 1) {
    std::stringstream kpca_projections_out_sstr;
    kpca_projections_out_sstr << vm["kpca_projections_out"].as<std::string>() <<
                              world.rank();
    arguments_out->kpca_projections_out_ = kpca_projections_out_sstr.str();
  }

  // Parse the maximum number of iterations.
  arguments_out->max_num_iterations_in_ = vm["max_num_iterations_in"].as<int>();
  if(world.rank() == 0) {
    std::cout << "Each Monte Carlo round will try up to " <<
              arguments_out->max_num_iterations_in_ << " rounds.\n";
  }

  // Parse the mode.
  arguments_out->mode_ = vm["mode"].as<std::string>();
  if(world.rank() == 0) {
    std::cout << "Running in the mode: " << arguments_out->mode_ << ".\n";
  }

  // Parse the number of KPCA components.
  arguments_out->num_kpca_components_in_ =
    (vm["mode"].as<std::string>() == "kde") ?
    1 : vm["num_kpca_components_in"].as<int>();
  if(world.rank() == 0) {
    std::cout << "Requesting " << arguments_out->num_kpca_components_in_ <<
              " kernel PCA components...\n";
  }

  // Parse whether the centering is requested for KPCA or not.
  arguments_out->do_centering_ = (vm.count("do_centering") > 0);
  if(world.rank() == 0 && arguments_out->mode_ == "kpca") {
    if(arguments_out->do_centering_) {
      std::cout << "Doing a centered kernel PCA.\n";
    }
    else {
      std::cout << "Doing a non-centered version of kernel PCA.\n";
    }
  }

  // Parse whether the naive mode is on or not.
  arguments_out->do_naive_ = (vm.count("do_naive") > 0 && world.size() == 1);
  if(world.rank() == 0) {
    if(arguments_out->do_naive_) {
      std::cout << "Computing naively as well.\n";
    }
    else {
      std::cout << "Just doing approximation.\n";
    }
  }

  // Parse the number of threads.
  arguments_out->num_threads_in_ = vm["num_threads_in"].as<int>();
  if(world.rank() == 0) {
    printf("Using %d threads for shared memory parallelism...\n",
           arguments_out->num_threads_in_);
  }
  return false;
}

bool DistributedKpcaArgumentParser::ConstructBoostVariableMap(
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

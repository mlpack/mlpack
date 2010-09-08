#ifndef MLPACK_KDE_KDE_DEV_H
#define MLPACK_KDE_KDE_DEV_H

#include "kde.h"

ml::Kde::TableType *ml::Kde::query_table() {
  return query_table_;
}

ml::Kde::TableType *ml::Kde::reference_table() {
  return reference_table_;
}

ml::Kde::GlobalType &ml::Kde::global() {
  return global_;
}

bool ml::Kde::is_monochromatic() const {
  return is_monochromatic_;
}

void ml::Kde::Compute(ml::Kde::ResultType *result_out) {

}

void ml::Kde::Init(ml::KdeArguments &arguments_in) {

  reference_table_ = arguments_in.reference_table_;
  if (arguments_in.query_table_ == NULL) {
    is_monochromatic_ = true;
    query_table_ = reference_table_;
  }
  else {
    is_monochromatic_ = false;
    query_table_ = arguments_in.query_table_;
  }

  // Declare the global constants.
  global_.Init(
    reference_table_, query_table_, arguments_in.bandwidth_, is_monochromatic_,
    arguments_in.relative_error_, arguments_in.probability_);
}

void ml::Kde::set_bandwidth(double bandwidth_in) {
  global_.set_bandwidth(bandwidth_in);
}

bool ml::Kde::ConstructBoostVariableMap_(
  const std::vector<std::string> &args,
  boost::program_options::variables_map *vm) {

  boost::program_options::options_description desc("Available options");
  desc.add_options()(
    "help", "Print this information."
  )(
    "references_in",
    boost::program_options::value<std::vector<std::string> >(),
    "REQUIRED file containing reference data."
  )(
    "queries_in",
    boost::program_options::value<std::string>(),
    "OPTIONAL file containing query positions.  If omitted, KDE computes "
    "the leave-one-out density at each reference point."
  )(
    "densities_out",
    boost::program_options::value<std::string>()->default_value(
      "densities_out.csv"),
    "OPTIONAL file to store computed densities."
  )(
    "kernel",
    boost::program_options::value<std::string>()->default_value("epan"),
    "Kernel function used by KDE.  One of:\n"
    "  epan, gaussian"
  )(
    "bandwidth",
    boost::program_options::value<double>(),
    "OPTIONAL kernel bandwidth, if you set --bandwidth_selection flag, "
    "then the --bandwidth will be ignored."
  )
  (
    "probability",
    boost::program_options::value<double>()->default_value(1.0),
    "Probability guarantee for the approximation of KDE."
  )(
    "relative_error",
    boost::program_options::value<double>()->default_value(0.01),
    "Relative error for the approximation of KDE."
  )(
    "leaf_size",
    boost::program_options::value<int>()->default_value(20),
    "Maximum number of points at a leaf of the tree.  More saves on tree "
    "overhead but too much hurts asymptotic run-time."
  );

  boost::program_options::command_line_parser clp(args);
  clp.style(boost::program_options::command_line_style::default_style
            ^ boost::program_options::command_line_style::allow_guessing);
  try {
    boost::program_options::store(clp.options(desc).run(), *vm);
  }
  catch (const boost::program_options::invalid_option_value &e) {
    std::cerr << "Invalid Argument: " << e.what();
    exit(0);
  }
  catch (const boost::program_options::invalid_command_line_syntax &e) {
    std::cerr << "Invalid command line syntax: " << e.what();
    exit(0);
  }
  catch (const boost::program_options::unknown_option &e) {
    std::cerr << "Unknown option: " << e.what() ;
    exit(0);
  }

  boost::program_options::notify(*vm);
  if (vm->count("help")) {
    std::cout << desc << "\n";
    return true;
  }

  // Validate the arguments. Only immediate dying is allowed here, the
  // parsing is done later.
  if (vm->count("references_in") == 0) {
    std::cerr << "Missing required --references_in";
    exit(0);
  }
  if ((*vm)["kernel"].as<std::string>() != "gaussian" &&
      (*vm)["kernel"].as<std::string>() != "epan") {
    std::cerr << "We support only epan or gaussian for the kernel.";
    exit(0);
  }
  if (vm->count("bandwidth") > 0 && (*vm)["bandwidth"].as<double>() <= 0) {
    std::cerr << "The --bandwidth requires a positive real number";
    exit(0);
  }
  if (vm->count("bandwidth") == 0) {
    std::cerr << "Missing required --bandwidth";
    exit(0);
  }
  if ((*vm)["probability"].as<double>() <= 0 ||
      (*vm)["probability"].as<double>() > 1) {
    std::cerr << "The --probability requires a real number $0 < p <= 1$ ";
    exit(0);
  }
  if ((*vm)["relative_error"].as<double>() < 0) {
    std::cerr << "The --relative_error requires a real number $r >= 0$";
    exit(0);
  }
  if ((*vm)["leaf_size"].as<int>() <= 0) {
    std::cerr << "The --leaf_size needs to be a positive integer.";
    exit(0);
  }
  return false;
}

void ml::Kde::ParseArguments(
  int argc,
  char *argv[],
  ml::KdeArguments *arguments_out) {

  // Convert C input to C++; skip executable name for Boost.
  std::vector<std::string> args(argv + 1, argv + argc);

  // Construct the Boost variable map.
  boost::program_options::variables_map vm;
  ConstructBoostVariableMap_(args, &vm);

  // Given the constructed boost variable map, parse each argument.

  // Parse the reference set.
  arguments_out->reference_table_ = new core::table::Table();
  arguments_out->reference_table_->Init(vm["references_in"].as<std::string>());

  // Parse the query set.
  if (vm.count("queries_in") > 0) {
    arguments_out->query_table_ =
      arguments_out->query_table_ = new core::table::Table();
    arguments_out->query_table_->Init(vm["queries_in"].as<std::string>());
  }

  // Parse the bandwidth.
  arguments_out->bandwidth_ = vm["bandwidth"].as<double>();

  // Parse the relative error.
  arguments_out->relative_error_ = vm["relative_error"].as<double>();

  // Parse the probability.
  arguments_out->probability_ = vm["probability"].as<double>();
}

#endif

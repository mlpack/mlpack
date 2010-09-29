/** @file nbody_simulator_dev.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef PHYSPACK_NBODY_SIMULATOR_NBODY_SIMULATOR_DEV_H
#define PHYSPACK_NBODY_SIMULATOR_NBODY_SIMULATOR_DEV_H

#include "core/metric_kernels/lmetric.h"
#include "core/gnp/tripletree_dfs_dev.h"
#include "nbody_simulator.h"

physpack::nbody_simulator::NbodySimulator::TableType *
physpack::nbody_simulator::NbodySimulator::table() {
  return table_;
}

physpack::nbody_simulator::NbodySimulator::GlobalType
&physpack::nbody_simulator::NbodySimulator::global() {
  return global_;
}

void physpack::nbody_simulator::NbodySimulator::Compute(
  const physpack::nbody_simulator::NbodySimulatorArguments &arguments_in,
  physpack::nbody_simulator::NbodySimulatorResult *result_out) {

  // Instantiate a dual-tree algorithm of the KDE.
  core::gnp::TripletreeDfs<physpack::nbody_simulator::NbodySimulator>
  tripletree_dfs;
  tripletree_dfs.Init(*this);

  // Compute the result.
  tripletree_dfs.Compute(* arguments_in.metric_, result_out);
}

void physpack::nbody_simulator::NbodySimulator::Init(
  physpack::nbody_simulator::NbodySimulatorArguments &arguments_in) {

  table_ = arguments_in.table_;

  // Declare the global constants.
  global_.Init(table_, arguments_in.relative_error_, arguments_in.probability_);
}

bool physpack::nbody_simulator::NbodySimulator::ConstructBoostVariableMap_(
  const std::vector<std::string> &args,
  boost::program_options::variables_map *vm) {

  boost::program_options::options_description desc("Available options");
  desc.add_options()(
    "help", "Print this information."
  )(
    "references_in",
    boost::program_options::value<std::string>(),
    "REQUIRED file containing reference data."
  )(
    "potentials_out",
    boost::program_options::value<std::string>()->default_value(
      "potentials_out.csv"),
    "OPTIONAL file to store computed potentials."
  )(
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
    "Maximum number of points at a leaf of the tree."
  );

  boost::program_options::command_line_parser clp(args);
  clp.style(boost::program_options::command_line_style::default_style
            ^ boost::program_options::command_line_style::allow_guessing);
  try {
    boost::program_options::store(clp.options(desc).run(), *vm);
  }
  catch (const boost::program_options::invalid_option_value &e) {
    std::cerr << "Invalid Argument: " << e.what() << "\n";
    exit(0);
  }
  catch (const boost::program_options::invalid_command_line_syntax &e) {
    std::cerr << "Invalid command line syntax: " << e.what() << "\n";
    exit(0);
  }
  catch (const boost::program_options::unknown_option &e) {
    std::cerr << "Unknown option: " << e.what() << "\n";
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
    std::cerr << "Missing required --references_in.\n";
    exit(0);
  }
  if ((*vm)["probability"].as<double>() <= 0 ||
      (*vm)["probability"].as<double>() > 1) {
    std::cerr << "The --probability requires a real number $0 < p <= 1$.\n";
    exit(0);
  }
  if ((*vm)["relative_error"].as<double>() < 0) {
    std::cerr << "The --relative_error requires a real number $r >= 0$.\n";
    exit(0);
  }
  if ((*vm)["leaf_size"].as<int>() <= 0) {
    std::cerr << "The --leaf_size needs to be a positive integer.\n";
    exit(0);
  }
  return false;
}

void physpack::nbody_simulator::NbodySimulator::ParseArguments(
  const std::vector<std::string> &args,
  physpack::nbody_simulator::NbodySimulatorArguments *arguments_out) {

  // A L2 metric to index the table to use.
  arguments_out->metric_ = new core::metric_kernels::LMetric<2>();

  // Construct the Boost variable map.
  boost::program_options::variables_map vm;
  ConstructBoostVariableMap_(args, &vm);

  // Given the constructed boost variable map, parse each argument.

  // Parse the densities out file.
  arguments_out->potentials_out_ = vm["potentials_out"].as<std::string>();

  // Parse the leaf size.
  arguments_out->leaf_size_ = vm["leaf_size"].as<int>();
  std::cout << "Using the leaf size of " << arguments_out->leaf_size_ << "\n";

  // Parse the reference set and index the tree.
  std::cout << "Reading in the reference set: " <<
            vm["references_in"].as<std::string>() << "\n";
  arguments_out->table_ = new core::table::Table();
  arguments_out->table_->Init(vm["references_in"].as<std::string>());
  std::cout << "Finished reading in the reference set.\n";
  std::cout << "Building the reference tree.\n";
  arguments_out->table_->IndexData(
    *(arguments_out->metric_), arguments_out->leaf_size_);
  std::cout << "Finished building the reference tree.\n";

  // Parse the relative error.
  arguments_out->relative_error_ = vm["relative_error"].as<double>();
  std::cout << "Relative error of " << arguments_out->relative_error_ << "\n";

  // Parse the probability.
  arguments_out->probability_ = vm["probability"].as<double>();
  std::cout << "Probability of " << arguments_out->probability_ << "\n";
}

void physpack::nbody_simulator::NbodySimulator::ParseArguments(
  int argc,
  char *argv[],
  physpack::nbody_simulator::NbodySimulatorArguments *arguments_out) {

  // Convert C input to C++; skip executable name for Boost.
  std::vector<std::string> args(argv + 1, argv + argc);

  ParseArguments(args, arguments_out);
}

#endif

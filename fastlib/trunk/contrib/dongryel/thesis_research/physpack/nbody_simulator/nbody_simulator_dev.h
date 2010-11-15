/** @file nbody_simulator_dev.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef PHYSPACK_NBODY_SIMULATOR_NBODY_SIMULATOR_DEV_H
#define PHYSPACK_NBODY_SIMULATOR_NBODY_SIMULATOR_DEV_H

#include "core/metric_kernels/lmetric.h"
#include "core/gnp/tripletree_dfs_dev.h"
#include "nbody_simulator.h"

template<typename TableType>
TableType *physpack::nbody_simulator::NbodySimulator<TableType>::table() {
  return table_;
}

template<typename TableType>
typename physpack::nbody_simulator::NbodySimulator<TableType>::GlobalType
&physpack::nbody_simulator::NbodySimulator<TableType>::global() {
  return global_;
}

template<typename TableType>
void physpack::nbody_simulator::NbodySimulator<TableType>::NaiveCompute(
  const physpack::nbody_simulator::NbodySimulatorArguments<TableType> &arguments_in,
  const physpack::nbody_simulator::NbodySimulatorResult &approx_result_in,
  physpack::nbody_simulator::NbodySimulatorResult *naive_result_out) {

  // Instantiate a dual-tree algorithm of the KDE.
  core::gnp::TripletreeDfs < physpack::nbody_simulator::NbodySimulator <
  TableType > > tripletree_dfs;
  tripletree_dfs.Init(*this);

  // Compute the result.
  core::util::Timer compute_timer;
  compute_timer.Start();
  tripletree_dfs.NaiveCompute(
    * arguments_in.metric_, naive_result_out);
  compute_timer.End();
  std::cout << compute_timer.GetTotalElapsedTime() << " seconds spent on "
            "the naive potential computation.\n";

  // Compute the deviation of the computed potentials. First, sort the
  // absolute values of the naively computed potentials.
  std::vector< std::pair<double, std::pair< double, double> > >
  naive_potential_copy;
  naive_potential_copy.resize(naive_result_out->potential_e_.size());
  for(unsigned int i = 0; i < naive_result_out->potential_e_.size(); i++) {
    naive_potential_copy[i].first = fabs(naive_result_out->potential_e_[i]);
    naive_potential_copy[i].second.first = naive_result_out->potential_e_[i];
    naive_potential_copy[i].second.second = approx_result_in.potential_e_[i];
  }
  std::sort(naive_potential_copy.begin(), naive_potential_copy.end());

  // Check whether the upper quantile of the sorted absolute
  // potentials satisfy the error bound.
  int satisfied_count = 0;
  int start_index = static_cast<int>(
                      floor(
                        naive_potential_copy.size() * arguments_in.summary_compute_quantile_));

  for(unsigned int i = start_index; i < naive_potential_copy.size(); i++) {
    if(fabs(
          naive_potential_copy[i].second.first -
          naive_potential_copy[i].second.second) <= arguments_in.relative_error_ *
        fabs(naive_potential_copy[i].second.first)) {
      satisfied_count++;
    }
  }
}

template<typename TableType>
void physpack::nbody_simulator::NbodySimulator<TableType>::Compute(
  const physpack::nbody_simulator::NbodySimulatorArguments<TableType> &arguments_in,
  physpack::nbody_simulator::NbodySimulatorResult *result_out) {

  // Instantiate a dual-tree algorithm of the KDE.
  core::gnp::TripletreeDfs < physpack::nbody_simulator::NbodySimulator <
  TableType > > tripletree_dfs;
  tripletree_dfs.Init(*this);

  // Compute the result.
  tripletree_dfs.Compute(* arguments_in.metric_, result_out);

  // Copy the number of prunes.
  result_out->num_deterministic_prunes_ =
    tripletree_dfs.num_deterministic_prunes();
  result_out->num_monte_carlo_prunes_ =
    tripletree_dfs.num_monte_carlo_prunes();
}

template<typename TableType>
void physpack::nbody_simulator::NbodySimulator<TableType>::Init(
  physpack::nbody_simulator::NbodySimulatorArguments<TableType> &arguments_in) {

  table_ = arguments_in.table_;

  // Declare the global constants.
  global_.Init(
    table_, arguments_in.relative_error_, arguments_in.probability_,
    arguments_in.summary_compute_quantile_);
}

template<typename TableType>
bool physpack::nbody_simulator::NbodySimulator<TableType>::ConstructBoostVariableMap_(
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
    "verify_accuracy", "Verify the accuracy of the computed potentials"
    " against the naive results."
  )(
    "summary_compute_quantile",
    boost::program_options::value<double>()->default_value(0.2),
    "OPTIONAL The quantile for computing summary results during computation."
  )(
    "probability",
    boost::program_options::value<double>()->default_value(1.0),
    "Probability guarantee for the approximation of KDE."
  )(
    "relative_error",
    boost::program_options::value<double>()->default_value(0.1),
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
  if(vm->count("references_in") == 0) {
    std::cerr << "Missing required --references_in.\n";
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
  return false;
}

template<typename TableType>
void physpack::nbody_simulator::NbodySimulator<TableType>::ParseArguments(
  const std::vector<std::string> &args,
  physpack::nbody_simulator::NbodySimulatorArguments<TableType> *arguments_out) {

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
  arguments_out->table_ = new TableType();
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

  // Parse the summary compute quantile.
  arguments_out->summary_compute_quantile_ =
    vm["summary_compute_quantile"].as<double>();
  std::cout << "Summary compute quantile of " <<
            arguments_out->summary_compute_quantile_ << "\n";

  // Determine whether we need to verify against the naive.
  arguments_out->verify_accuracy_ = (vm.count("verify_accuracy") > 0);
}

template<typename TableType>
void physpack::nbody_simulator::NbodySimulator<TableType>::ParseArguments(
  int argc,
  char *argv[],
  physpack::nbody_simulator::NbodySimulatorArguments<TableType> *arguments_out) {

  // Convert C input to C++; skip executable name for Boost.
  std::vector<std::string> args(argv + 1, argv + argc);

  ParseArguments(args, arguments_out);
}

#endif

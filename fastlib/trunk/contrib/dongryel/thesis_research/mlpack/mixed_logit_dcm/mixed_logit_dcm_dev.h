/** @file mixed_logit_dcm_dev.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_MIXED_LOGIT_DCM_MIXED_LOGIT_DCM_DEV_H
#define MLPACK_MIXED_LOGIT_DCM_MIXED_LOGIT_DCM_DEV_H

#include "mlpack/mixed_logit_dcm/mixed_logit_dcm.h"

namespace mlpack {
namespace mixed_logit_dcm {

template<typename TableType>
void MixedLogitDCM<TableType>::Init(
  mlpack::mixed_logit_dcm::MixedLogitDCMArguments <
  TableType > &arguments_in) {

  // Initialize the table for storing/accessing the attribute vector
  // for each person.
  table_.Init(arguments_in);
}

template<typename TableType>
void MixedLogitDCM<TableType>::Compute(
  const mlpack::mixed_logit_dcm::MixedLogitDCMArguments <
  TableType > &arguments_in,
  mlpack::mixed_logit_dcm::MixedLogitDCMResult *result_out) {

  static const double C = 1.04;
  static const double eta = 0.05;

  // The maximum average integration sample size.
  static const int R_MAX = 1000;

  // Here is the main entry of the algorithm.
  int num_data_samples =
    static_cast<int>(
      table_.num_people() *
      arguments_in.initial_dataset_sample_rate_);
  std::vector<int> num_integration_samples(
    num_data_samples, std::max(
      static_cast<int>(arguments_in.initial_integration_sample_rate_ * R_MAX),
      36));

  // Compute the initial simulated log-likelihood, the gradient, and
  // the Hessian.
  double current_simulated_log_likelihood = table_.SimulatedLogLikelihood();
  core::table::DensePoint current_gradient;
  core::table::DenseMatrix current_hessian;

  for(int iter = 0; ; iter++) {


  }
}

template<typename TableType>
bool MixedLogitDCM<TableType>::ConstructBoostVariableMap_(
  const std::vector<std::string> &args,
  boost::program_options::variables_map *vm) {

  boost::program_options::options_description desc("Available options");
  desc.add_options()(
    "help", "Print this information."
  )(
    "attributes_in",
    boost::program_options::value<std::string>(),
    "REQUIRED file containing the vector of attributes."
  )("num_discrete_choices_per_person_in",
    boost::program_options::value<std::string>(),
    "REQUIRED The number of alternatives per each person."
   )(
     "predictions_out",
     boost::program_options::value<std::string>()->default_value(
       "densities_out.csv"),
     "OPTIONAL file to store the predicted discrete choices."
   )("initial_dataset_sample_rate",
     boost::program_options::value<double>()->default_value(0.1),
     "OPTIONAL the rate at which to sample the entire dataset in the "
     "beginning."
    )("initial_integration_sample_rate",
      boost::program_options::value<double>()->default_value(0.01),
      "OPTIONAL The percentage of the maximum average integration sample "
      "to start with."
     )("model_out",
       boost::program_options::value<std::string>()->default_value(
         "model_out.csv"),
       "file to output the computed discrete choice model."
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

  // Validate the arguments. Only immediate quitting is allowed here,
  // the parsing is done later.
  if(vm->count("attributes_in") == 0) {
    std::cerr << "Missing required --attributes_in.\n";
    exit(0);
  }
  if(vm->count("num_discrete_choices_per_person_in") == 0) {
    std::cerr << "Missing required --num_discrete_choices_per_person_in.\n";
    exit(0);
  }
  return false;
}

template<typename TableType>
void MixedLogitDCM<TableType>::ParseArguments(
  const std::vector<std::string> &args,
  MixedLogitDCMArguments<TableType> *arguments_out) {

  // Construct the Boost variable map.
  boost::program_options::variables_map vm;
  if(ConstructBoostVariableMap_(args, &vm)) {
    exit(0);
  }

  // Given the constructed boost variable map, parse each argument.

  // Parse the set of attribute vectors.
  std::cout << "Reading in the attribute set: " <<
            vm["attributes_in"].as<std::string>() << "\n";
  arguments_out->attribute_table_ = new TableType();
  arguments_out->attribute_table_->Init(vm["attributes_in"].as<std::string>());
  std::cout << "Finished reading in the attributes set.\n";

  // Parse the number of discrete choices per each point.
  std::cout << "Reading in the number of discrete choices per each person: " <<
            vm["num_discrete_choices_per_person_in"].as<std::string>() << "\n";
  arguments_out->num_discrete_choices_per_person_ = new TableType();
  arguments_out->num_discrete_choices_per_person_->Init(
    vm["num_discrete_choices_per_person_in"].as<std::string>());

  // Parse the initial dataset sample rate.
  arguments_out->initial_dataset_sample_rate_ =
    vm[ "initial_dataset_sample_rate" ].as<double>();

  // Parse the initial integration sample rate;
  arguments_out->initial_integration_sample_rate_ =
    vm[ "initial_integration_sample_rate" ].as<double>();

  // Parse where to output the discrete choice model predictions.
  arguments_out->predictions_out_ = vm[ "predictions_out" ].as<std::string>();

  // Parse the output for the mixed logit discrete choice model.
  arguments_out->model_out_ = vm[ "model_out" ].as<std::string>();

  // The number of parameters that generate each $\beta$ is fixed now
  // as the Gaussian example in Appendix.
  // arguments_out->distribution_ = ;
}

template<typename TableType>
void MixedLogitDCM<TableType>::ParseArguments(
  int argc,
  char *argv[],
  MixedLogitDCMArguments<TableType> *arguments_out) {

  // Convert C input to C++; skip executable name for Boost.
  std::vector<std::string> args(argv + 1, argv + argc);

  ParseArguments(args, arguments_out);
}
};
};

#endif

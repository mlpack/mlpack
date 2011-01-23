/** @file mixed_logit_dcm_dev.h
 *
 *  The implementation of mixed logit discrete choice model.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_MIXED_LOGIT_DCM_MIXED_LOGIT_DCM_DEV_H
#define MLPACK_MIXED_LOGIT_DCM_MIXED_LOGIT_DCM_DEV_H

#include "mlpack/mixed_logit_dcm/mixed_logit_dcm.h"
#include "core/optimization/trust_region_dev.h"

namespace mlpack {
namespace mixed_logit_dcm {

template<typename TableType>
double MixedLogitDCM<TableType>::GradientErrorSecondPart_(
  const SamplingType &sample) const {

  // The temporary vector for extracting choice probability and its
  // gradient for the second error component. Again, careful aliasing
  // is done here.
  arma::vec second_tmp_vector;
  second_tmp_vector.set_size(2 *(table_->num_parameters() + 1));
  arma::vec second_tmp_choice_probability_gradient_outer(
    second_tmp_vector.memptr() + 1, table_->num_parameters(), false);
  arma::vec second_tmp_choice_probability_gradient_inner(
    second_tmp_vector.memptr() + table_->num_parameters() + 2,
    table_->num_parameters(), false);

  double second_part = 0;
  for(int i = 0; i < table_->num_people(); i++) {

    // Get the person index.
    int person_index = table_->shuffled_indices_for_person(i);

    // Get the simulated choice probability and the simulated choice
    // probability gradient for the given/ person.
    double simulated_choice_probability =
      sample.simulated_choice_probability(person_index);
    arma::vec simulated_choice_probability_gradient;
    sample.simulated_choice_probability_gradient(
      person_index, &simulated_choice_probability_gradient);

    // The integration samples for the given person.
    const std::vector< arma::vec > &integration_samples =
      table_->integration_samples(person_index);
    double normalization_factor =
      1.0 / static_cast<double>(
        integration_samples.size() * (integration_samples.size() - 1));

    // Loop through each integration sample.
    for(unsigned int j = 0; j < integration_samples.size(); j++) {
      const arma::vec &integration_sample = integration_samples[j];
      double choice_probability =
        table_->choice_probability(person_index, integration_sample);
      table_->choice_probability_gradient(
        person_index, &first_tmp_choice_probability_gradient);

      // Take the dot product between the two vectors and square it.
      first_part +=
        normalization_factor *
        core::math::Sqr(arma::dot(delta_hii, first_temp_vector));
    }
  }

  // Multiply the second part by 4.
  second_part *= 4.0;
  return second_part;
}

template<typename TableType>
double MixedLogitDCM<TableType>::GradientErrorFirstPart_(
  const SamplingType &sample) const {

  // The temporary vector for extracting choice probability and its
  // gradient. Careful aliasing is done here.
  arma::vec first_tmp_vector;
  first_tmp_vector.set_size(table_->num_parameters() + 1);
  arma::vec first_tmp_choice_probability_gradient(
    first_tmp_vector.memptr() + 1, table_->num_parameters(), false);

  double first_part = 0;
  for(int i = 0; i < table_->num_people(); i++) {

    // Get the person index.
    int person_index = table_->shuffled_indices_for_person(i);

    // Get the simulated choice probability and the simulated choice
    // probability gradient for the given/ person.
    double simulated_choice_probability =
      sample.simulated_choice_probability(person_index);
    arma::vec simulated_choice_probability_gradient;
    sample.simulated_choice_probability_gradient(
      person_index, &simulated_choice_probability_gradient);

    // First form the $\Delta h_ii vector.
    arma::vec delta_hii;
    delta_hii.set_size(simulated_choice_probability_gradient.n_elem + 1);
    delta_hii[0] = -2.0 / core::math::Pow<3, 1>(simulated_choice_probability) *
                   arma::dot(
                     simulated_choice_probability_gradient,
                     simulated_choice_probability_gradient);
    delta_hii.submat(1, delta_hii.n_elem - 1, 0, 0) =
      2.0 / core::math::Sqr(simulated_choice_probability) *
      simulated_choice_probability_gradient;

    // The integration samples for the given person.
    const std::vector< arma::vec > &integration_samples =
      table_->integration_samples(person_index);
    double normalization_factor =
      1.0 / static_cast<double>(
        integration_samples.size() * (integration_samples.size() - 1));

    // Loop through each integration sample.
    for(unsigned int j = 0; j < integration_samples.size(); j++) {
      const arma::vec &integration_sample = integration_samples[j];
      double choice_probability =
        table_->choice_probability(person_index, integration_sample);
      table_->choice_probability_gradient(
        person_index, &first_tmp_choice_probability_gradient);

      // Take the dot product between the two vectors and square it.
      first_part +=
        normalization_factor *
        core::math::Sqr(arma::dot(delta_hii, first_temp_vector));
    }
  }
  return first_part;
}

template<typename TableType>
double MixedLogitDCM<TableType>::GradientError_(
  const SamplingType &sample) const {

  // Compute the first part of the gradient error.
  double first_part = GradientErrorFirstPart_(sample);
  double second_part = GradientErrorSecondPart_(sample);

  // Add the two errors.
  gradient_error = first_part + second_part;

  // Divide by the normalization term, which is the total number of
  // people in the dataset raised to the 4-th power.
  gradient_error /=
    static_cast<double>(core::math::Pow<4, 1>(table_->num_people()));
  return gradient_error;
}

template<typename TableType>
double MixedLogitDCM<TableType>::SimulationError_(
  const SamplingType &first_sample,
  const SamplingType &second_sample) const {

  // Assumption: num_active_people in both samples are equal.
  double simulation_error = 0;

  // Lastly divide by squared of the number of active people.
  simulation_error /=
    core::math::Sqr(static_cast<double>(first_sample.num_active_people()));
  return simulation_error;
}

template<typename TableType>
double MixedLogitDCM<TableType>::SampleDataError_(
  const SamplingType &first_sample,
  const SamplingType &second_sample) const {

  // Assumption: num_active_people in both samples are equal.
  double correction_factor =
    static_cast<double>(
      table_.num_people() - first_sample.num_active_people()) /
    static_cast<double>(table_.num_people() - 1);
  double factor = correction_factor /
                  static_cast<double>(
                    first_sample.num_active_people() *
                    (first_sample.num_active_people() - 1));

  // Compute the average difference of simulated log probabilities.
  double average_difference = 0;
  for(int i = 0; i < first_sample.num_active_people(); i++) {
    int person_index = table_.shuffled_indices_for_person(i);
    average_difference +=
      (log(first_sample.simulated_choice_probability(person_index)) -
       log(second_sample.simulated_choice_probability(person_index)));
  }
  average_difference /= static_cast<double>(first_sample.num_active_people());
  double variance = 0;

  // Given the average difference, compute the variance.
  for(int i = 0; i < first_sample.num_active_people(); i++) {
    int person_index = table_.shuffled_indices_for_person(i);
    double difference =
      log(first_sample.simulated_choice_probability(person_index)) -
      log(second_sample.simulated_choice_probability(person_index));
    variance += core::math::Sqr(difference - average_difference);
  }
  return factor * variance;
}

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

  // The maximum average integration sample size.
  static const int R_MAX = 1000;

  // Here is the main entry of the algorithm.
  int num_data_samples =
    static_cast<int>(
      table_.num_people() *
      arguments_in.initial_dataset_sample_rate_);
  int num_integration_samples =
    std::max(
      static_cast<int>(arguments_in.initial_integration_sample_rate_ * R_MAX),
      36);

  // Compute the initial simulated log-likelihood, the gradient, and
  // the Hessian.
  double current_simulated_log_likelihood = 0;//table_.SimulatedLogLikelihood();
  arma::vec current_gradient;
  arma::mat current_hessian;
  //table_.SimulatedLoglikelihoodGradient(&current_gradient);
  //table_.SimulatedLoglikelihoodHessian(&current_hessian);

  // Initialize the starting optimization parameter $\theta_0$ and its
  // associated sampling information.
  typedef mlpack::mixed_logit_dcm::DCMTable<TableType> DCMTableType;
  mlpack::mixed_logit_dcm::MixedLogitDCMSampling<DCMTableType>
  iterate_sampling;
  iterate_sampling.Init(
    &table_, num_data_samples, num_integration_samples);


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
      )("trust_region_search_method",
        boost::program_options::value<std::string>()->default_value("cauchy"),
        "OPTIONAL Trust region search method.  One of:\n"
        "  cauchy, dogleg, steihaug"
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

  // Parse how to perform the trust region search.
  arguments_out->trust_region_search_method_ =
    vm[ "trust_region_search_method" ].as<std::string>();

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
}
}

#endif

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
  second_tmp_vector.set_size(2 *(table_.num_parameters() + 1));
  arma::vec second_tmp_choice_probability_gradient_outer(
    second_tmp_vector.memptr() + 1, table_.num_parameters(), false);
  arma::vec second_tmp_choice_probability_gradient_inner(
    second_tmp_vector.memptr() + table_.num_parameters() + 2,
    table_.num_parameters(), false);

  // Temporary vector.
  arma::vec outer_choice_probabilities;
  arma::vec inner_choice_probabilities;
  arma::vec outer_choice_prob_weighted_attribute_vector;
  arma::vec inner_choice_prob_weighted_attribute_vector;

  // The quantity to be eventually returned.
  double second_part = 0;

  // The outer sum loop.
  for(int i = 0; i < table_.num_people(); i++) {

    // Get the outer person index and its discrete choice index.
    int outer_person_index = table_.shuffled_indices_for_person(i);
    int outer_discrete_choice_index =
      table_.get_discrete_choice_index(outer_person_index);

    // Get the simulated choice probability and the simulated choice
    // probability gradient for the given outer person.
    double outer_simulated_choice_probability =
      sample.simulated_choice_probability(outer_person_index);
    arma::vec outer_simulated_choice_probability_gradient;
    sample.simulated_choice_probability_gradient(
      outer_person_index, &outer_simulated_choice_probability_gradient);

    // The integration samples for the given outer person.
    const std::vector< arma::vec > &integration_samples =
      sample.integration_samples(outer_person_index);
    double normalization_factor =
      1.0 / static_cast<double>(
        integration_samples.size() * (integration_samples.size() - 1));

    // The inner sum loop.
    for(int k = i + 1; k < table_.num_people(); k++) {

      // Get the inner person index and its discrete choice.
      int inner_person_index = table_.shuffled_indices_for_person(k);
      int inner_discrete_choice_index =
        table_.get_discrete_choice_index(inner_person_index);

      // Get the simulated choice probability and the simulated choice
      // probability gradient for the given inner person.
      double inner_simulated_choice_probability =
        sample.simulated_choice_probability(inner_person_index);
      arma::vec inner_simulated_choice_probability_gradient;
      sample.simulated_choice_probability_gradient(
        inner_person_index, &inner_simulated_choice_probability_gradient);

      // Compute delta_hik (Equation 2.3).
      arma::vec delta_hik;
      delta_hik.set_size(2 *(table_.num_parameters() + 1));
      double first_factor =
        -1.0 / (
          core::math::Sqr(outer_simulated_choice_probability) *
          inner_simulated_choice_probability);
      double second_factor =
        1.0 / (
          outer_simulated_choice_probability *
          inner_simulated_choice_probability);
      double third_factor =
        -1.0 / (
          core::math::Sqr(inner_simulated_choice_probability) *
          outer_simulated_choice_probability);
      double dot_product =
        arma::dot(
          outer_simulated_choice_probability_gradient,
          inner_simulated_choice_probability_gradient);
      delta_hik[0] = first_factor * dot_product;
      delta_hik.submat(1, table_.num_parameters(), 0, 0) =
        second_factor * outer_simulated_choice_probability_gradient;
      delta_hik[table_.num_parameters() + 1] = third_factor * dot_product;
      delta_hik.submat(
        table_.num_parameters() + 2, delta_hik.n_elems - 1, 0, 0) =
          second_factor * inner_simulated_choice_probability_gradient;

      // Loop through each integration sample.
      for(unsigned int j = 0; j < integration_samples.size(); j++) {
        const arma::vec &integration_sample = integration_samples[j];

        // Fill out the upper half of the covariance expression in the
        // right hand side of Equation 2.3
        table_.choice_probabilities(
          outer_person_index, integration_sample, &outer_choice_probabilities);
        second_tmp_vector[0] =
          outer_choice_probabilities[ outer_discrete_choice_index ];
        table_.distribution()->ChoiceProbabilityWeightedAttributeVector(
          table_, outer_person_index, outer_choice_probabilities,
          &outer_choice_prob_weighted_attribute_vector);
        table_.distribution()->ChoiceProbabilityGradientWithRespectToParameter(
          sample.parameters(), table_,
          outer_person_index, outer_choice_probabilities,
          outer_choice_prob_weighted_attribute_vector,
          &second_tmp_choice_probability_gradient_outer);

        // Fill out the lower half.
        table_.choice_probabilities(
          inner_person_index, integration_sample, &inner_choice_probabilities);
        second_tmp_vector[ table_.num_parameters() + 1] =
          inner_choice_probabilities[ inner_discrete_choice_index ];
        table_.distribution()->ChoiceProbabilityWeightedAttributeVector(
          table_, inner_person_index, inner_choice_probabilities,
          &inner_choice_prob_weighted_attribute_vector);
        table_.distribution()->ChoiceProbabilityGradientWithRespectToParameter(
          sample.parameters(), table_,
          inner_person_index, inner_choice_probabilities,
          inner_choice_prob_weighted_attribute_vector,
          &second_tmp_choice_probability_gradient_inner);

        // Take the dot product between the two vectors and square it.
        second_part +=
          normalization_factor *
          core::math::Sqr(arma::dot(delta_hik, second_tmp_vector));
      }
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
  first_tmp_vector.set_size(table_.num_parameters() + 1);
  arma::vec first_tmp_choice_probability_gradient(
    first_tmp_vector.memptr() + 1, table_.num_parameters(), false);

  // The choice probability vector (temporary space) and the attribute
  // vector weighted by it.
  arma::vec choice_probabilities;
  arma::vec choice_prob_weighted_attribute_vector;

  double first_part = 0;
  for(int i = 0; i < table_.num_people(); i++) {

    // Get the person index and its discrete choice.
    int person_index = table_.shuffled_indices_for_person(i);
    int discrete_choice_index = table_.get_discrete_choice_index(person_index);

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
      sample.integration_samples(person_index);
    double normalization_factor =
      1.0 / static_cast<double>(
        integration_samples.size() * (integration_samples.size() - 1));

    // Loop through each integration sample.
    for(unsigned int j = 0; j < integration_samples.size(); j++) {
      const arma::vec &integration_sample = integration_samples[j];

      // Get the choice probability vector and its weighted version.
      table_.choice_probabilities(
        person_index, integration_sample, &choice_probabilities);
      table_.distribution()->ChoiceProbabilityWeightedAttributeVector(
        table_, person_index, choice_probabilities,
        &choice_prob_weighted_attribute_vector);
      first_tmp_vector[0] = choice_probabilities[discrete_choice_index];
      table_.distribution()->ChoiceProbabilityGradientWithRespectToParameter(
        sample.parameters(), table_, person_index,
        choice_probabilities, choice_prob_weighted_attribute_vector,
        &first_tmp_choice_probability_gradient);

      // Take the dot product between the two vectors and square it.
      first_part +=
        normalization_factor *
        core::math::Sqr(arma::dot(delta_hii, first_tmp_vector));
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
  double gradient_error = first_part + second_part;

  // Divide by the normalization term, which is the total number of
  // people in the dataset raised to the 4-th power.
  gradient_error /=
    static_cast<double>(core::math::Pow<4, 1>(table_.num_people()));
  return gradient_error;
}

template<typename TableType>
double MixedLogitDCM<TableType>::IntegrationSampleError_(
  const SamplingType &first_sample,
  const SamplingType &second_sample) const {

  // Assumption: num_active_people in both samples are equal.
  double simulation_error = 0;

  // Loop over each active people.
  for(int i = 0; i < first_sample.num_active_people(); i++) {

    // Get the active person index.
    int person_index = table_.shuffled_indices_for_person(i);

    // Get the integration samples for both samples.
    const std::vector< arma::vec > &first_integration_samples =
      first_sample.integration_samples(person_index);
    const std::vector< arma::vec > &second_integration_samples =
      second_sample.integration_samples(person_index);

    // Get the simulated choice probabilities.
    double first_simulated_choice_probability =
      first_sample.simulated_choice_probability(person_index);
    double second_simulated_choice_probability =
      second_sample.simulated_choice_probability(person_index);

    // Accumulate the difference.
    core::monte_carlo::MeanVariancePair difference_mean_variance;
    for(unsigned j = 0; j < first_integration_samples.size(); j++) {
      const arma::vec &first_integration_sample =
        first_integration_samples[j];
      const arma::vec &second_integration_sample =
        second_integration_samples[j];
      double first_choice_probability =
        table_.choice_probability(person_index, first_integration_sample);
      double second_choice_probability =
        table_.choice_probability(person_index, second_integration_sample);
      double difference =
        first_choice_probability / first_simulated_choice_probability -
        second_choice_probability / second_simulated_choice_probability;
      difference_mean_variance.push_back(difference);
    }
    simulation_error += difference_mean_variance.sample_mean_variance();
  }

  // Lastly divide by squared of the number of active people.
  simulation_error /=
    core::math::Sqr(static_cast<double>(first_sample.num_active_people()));
  return simulation_error;
}

template<typename TableType>
double MixedLogitDCM<TableType>::DataSampleError_(
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
  const ArgumentType &arguments_in,
  mlpack::mixed_logit_dcm::MixedLogitDCMResult *result_out) {

  // Here is the main entry of the algorithm.
  int initial_num_data_samples =
    static_cast<int>(
      table_.num_people() *
      arguments_in.initial_dataset_sample_rate_);
  int initial_num_integration_samples =
    std::max(
      static_cast<int>(
        arguments_in.initial_integration_sample_rate_ *
        arguments_in.max_num_integration_samples_per_person_), 36);

  // Initialize the starting optimization parameter $\theta_0$ and its
  // associated sampling information.
  arma::vec gradient;
  arma::mat hessian;
  SamplingType *iterate = new SamplingType();
  iterate->Init(
    &table_, initial_num_data_samples, initial_num_integration_samples);
  iterate->parameters().zeros(table_.num_parameters());
  iterate->NegativeSimulatedLogLikelihoodGradient(&gradient);
  iterate->NegativeSimulatedLogLikelihoodHessian(&hessian);

  // The step direction and its 2-norm.
  arma::vec p;
  double p_norm;

  // The initial trust region radius is set to the 10 % of the maximum
  // trust region radius.
  double current_radius = 0.1 * arguments_in.max_trust_region_radius_;

  // Enter the trust region loop.
  do {

    // Obtain the step direction by solving Equation 4.3
    // approximately.
    core::optimization::TrustRegionUtil::ObtainStepDirection(
      arguments_in.trust_region_search_method_, current_radius, gradient,
      hessian, &p, &p_norm);

    // Get the reduction ratio rho (Equation 4.4)
    SamplingType *next_iterate = new SamplingType();
    next_iterate->parameters() = iterate->parameters() + p;
    double iterate_function_value = iterate->NegativeSimulatedLogLikelihood();
    double next_iterate_function_value =
      next_iterate->NegativeSimulatedLogLikelihood();
    double decrease_predicted_by_model;
    double model_reduction_ratio =
      core::optimization::TrustRegionUtil::ReductionRatio(
        p, iterate_function_value, next_iterate_function_value,
        gradient, hessian, &decrease_predicted_by_model);

    // Compute the data sample error and the integration sample error.
    double data_sample_error = this->DataSampleError_(*iterate, *next_iterate);
    double integration_sample_error =
      this->IntegrationSampleError_(*iterate, *next_iterate);

    // Determine whether the termination condition has been reached.
    if(TerminationConditionReached_(
          arguments_in, decrease_predicted_by_model,
          data_sample_error, integration_sample_error,
          *iterate, gradient)) {
      break;
    }

    // If we are not ready to terminate yet, then consider one of the
    // three options. (1) Increase the data sample size; (2) Increase
    // the integration sample size; (3) Do a step to the new iterate.

    // Increase the data sample size.
    if(iterate->num_active_people() < table_.num_people() &&
        sqrt(data_sample_error) >= sqrt(integration_sample_error) &&
        fabs(decrease_predicted_by_model) <
        arguments_in.gradient_norm_threshold_ * sqrt(data_sample_error)) {

      int max_allowable_people = table_.num_people() -
                                 iterate->num_active_people();
      int predicted_increase =
        core::math::Sqr(
          arguments_in.gradient_norm_threshold_ *
          sqrt(data_sample_error) / decrease_predicted_by_model) *
        iterate->num_active_people() - iterate->num_active_people();
      int num_additional_people =
        std::min(std::max(1, predicted_increase), max_allowable_people);
      iterate->AddActivePeople(
        num_additional_people, initial_num_integration_samples);

      // Update the gradient and the hessian.
      iterate->NegativeSimulatedLogLikelihoodGradient(&gradient);
      iterate->NegativeSimulatedLogLikelihoodHessian(&hessian);

      // Delete the discarded iterate.
      delete next_iterate;
      continue;
    }

    // Increase the integration sample size.
    if(sqrt(integration_sample_error) > sqrt(data_sample_error) &&
        fabs(decrease_predicted_by_model) <
        arguments_in.gradient_norm_threshold_ *
        sqrt(integration_sample_error)) {

      // Update the gradient and the hessian.
      iterate->NegativeSimulatedLogLikelihoodGradient(&gradient);
      iterate->NegativeSimulatedLogLikelihoodHessian(&hessian);

      // Delete the discarded iterate.
      delete next_iterate;
      continue;
    }

    // Now the third case: adjust the trust region radius and make the
    // next iterate based on the trust region optimization.
    if(model_reduction_ratio < 0.10) {
      current_radius = p_norm / 2.0;
    }
    else {
      if(model_reduction_ratio > 0.80 &&
          fabs(p_norm - current_radius) <= 0.001) {
        current_radius = 2.0 * current_radius;
      }
    }
    const double eta = 0.05;
    if(model_reduction_ratio > eta) {

      // Accept the next iterate.
      delete iterate;
      iterate = next_iterate;

      // Update the gradient and the hessian.
      iterate->NegativeSimulatedLogLikelihoodGradient(&gradient);
      iterate->NegativeSimulatedLogLikelihoodHessian(&hessian);
    }
    else {

      // Delete the discarded iterate.
      delete next_iterate;
    }
  }
  while(true);
}

template<typename TableType>
bool MixedLogitDCM<TableType>::TerminationConditionReached_(
  const ArgumentType &arguments_in,
  double predicted_objective_value_improvement, double data_sample_error,
  double integration_sample_error, const SamplingType &sampling,
  const arma::vec &gradient) const {

  // Termination condition is only considered when we use all the
  // people in the sampling (outer term consists of all people).
  if(sampling.num_active_people() == table_.num_people()) {

    // If the predicted improvement in the objective value is less
    // than the integration sample error and the integration sample
    // error is small,
    if(predicted_objective_value_improvement <
        arguments_in.gradient_norm_threshold_ * sqrt(integration_sample_error) &&
        sqrt(integration_sample_error) <
        arguments_in.integration_sample_error_threshold_) {

      // Compute the gradient error.
      double gradient_error = this->GradientError_(sampling);

      if(arma::dot(gradient, gradient) +
          arguments_in.gradient_norm_threshold_ *
          sqrt(gradient_error) <= 0.001) {
        return true;
      }
    }
  }
  return false;
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
  )(
    "num_discrete_choices_per_person_in",
    boost::program_options::value<std::string>(),
    "REQUIRED The number of alternatives per each person."
  )(
    "predictions_out",
    boost::program_options::value<std::string>()->default_value(
      "densities_out.csv"),
    "OPTIONAL file to store the predicted discrete choices."
  )(
    "initial_dataset_sample_rate",
    boost::program_options::value<double>()->default_value(0.1),
    "OPTIONAL the rate at which to sample the entire dataset in the "
    "beginning."
  )(
    "initial_integration_sample_rate",
    boost::program_options::value<double>()->default_value(0.01),
    "OPTIONAL The percentage of the maximum average integration sample "
    "to start with."
  )(
    "gradient_norm_threshold",
    boost::program_options::value<double>()->default_value(0.5),
    "OPTIONAL The threshold for determining the termination condition based "
    "on the gradient norm."
  )(
    "max_num_integration_samples_per_person",
    boost::program_options::value<int>()->default_value(1000),
    "OPTIONAL The maximum number of integration samples allowed per person."
  )(
    "integration_sample_error_threshold",
    boost::program_options::value<double>()->default_value(1e-9),
    "OPTIONAL The threshold for determining whether the integration sample "
    "error is small or not."
  )(
    "max_trust_region_radius",
    boost::program_options::value<double>()->default_value(10.0),
    "OPTIONAL The maximum trust region radius used in the trust region "
    "search."
  )(
    "trust_region_search_method",
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
  if((*vm)[ "trust_region_search_method" ].as<std::string>() != "cauchy" &&
      (*vm)[ "trust_region_search_method" ].as<std::string>() != "dogleg" &&
      (*vm)[ "trust_region_search_method" ].as<std::string>() != "steihaug") {
    std::cerr << "Invalid option specified for --trust_region_search_method.\n";
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

  // Parse the initial integration sample rate.
  arguments_out->initial_integration_sample_rate_ =
    vm[ "initial_integration_sample_rate" ].as<double>();

  // Parse the gradient norm threshold.
  arguments_out->gradient_norm_threshold_ =
    vm[ "gradient_norm_threshold" ].as<double>();

  // Parse the maximum number of integration samples per person.
  arguments_out->max_num_integration_samples_per_person_ =
    vm[ "max_num_integration_samples_per_person" ].as<int>();

  // Parse the integration sample error thresold.
  arguments_out->integration_sample_error_threshold_ =
    vm["integration_sample_error_threshold"].as<double>();

  // Parse where to output the discrete choice model predictions.
  arguments_out->predictions_out_ = vm[ "predictions_out" ].as<std::string>();

  // Parse the maximum trust region radius.
  arguments_out->max_trust_region_radius_ =
    vm[ "max_trust_region_radius" ].as<double>();

  // Parse how to perform the trust region search: cauchy, dogleg, steihaug
  if(vm[ "trust_region_search_method" ].as<std::string>() == "cauchy") {
    arguments_out->trust_region_search_method_ =
      core::optimization::TrustRegionSearchMethod::CAUCHY;
  }
  else if(vm[ "trust_region_search_method" ].as<std::string>() == "dogleg") {
    arguments_out->trust_region_search_method_ =
      core::optimization::TrustRegionSearchMethod::DOGLEG;
  }
  else {
    arguments_out->trust_region_search_method_ =
      core::optimization::TrustRegionSearchMethod::STEIHAUG;
  }

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

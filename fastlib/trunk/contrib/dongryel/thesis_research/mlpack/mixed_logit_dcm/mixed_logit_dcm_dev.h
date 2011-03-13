/** @file mixed_logit_dcm_dev.h
 *
 *  The implementation of mixed logit discrete choice model.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_MIXED_LOGIT_DCM_MIXED_LOGIT_DCM_DEV_H
#define MLPACK_MIXED_LOGIT_DCM_MIXED_LOGIT_DCM_DEV_H

#include "mlpack/mixed_logit_dcm/mixed_logit_dcm.h"
#include "core/optimization/quasi_newton_hessian_update.h"
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
      delta_hik.submat(
        arma::span(1, table_.num_parameters()), arma::span(0, 0)) =
          second_factor * outer_simulated_choice_probability_gradient;
      delta_hik[table_.num_parameters() + 1] = third_factor * dot_product;
      delta_hik.submat(
        arma::span(table_.num_parameters() + 2, delta_hik.n_elem - 1),
        arma::span(0, 0)) =
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
          outer_person_index, integration_sample,
          outer_choice_probabilities,
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
          inner_person_index, integration_sample,
          inner_choice_probabilities,
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
    delta_hii.submat(arma::span(1, delta_hii.n_elem - 1), arma::span(0, 0)) =
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
        sample.parameters(), table_, person_index, integration_sample,
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
void MixedLogitDCM<TableType>::IntegrationSampleErrorPerPerson_(
  int person_index,
  const SamplingType &first_sample,
  const SamplingType &second_sample,
  core::monte_carlo::MeanVariancePair *integration_sample_error) const {

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
  integration_sample_error->SetZero();
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
    integration_sample_error->push_back(difference);
  }
}

template<typename TableType>
double MixedLogitDCM<TableType>::IntegrationSampleError_(
  const SamplingType &first_sample,
  const SamplingType &second_sample) const {

  // Assumption: num_active_people in both samples are equal.
  double simulation_error = 0;

  // Loop over each active people.
  for(int i = 0; i < first_sample.num_active_people(); i++) {

    // Get the active person index and the corresponding integration
    // sample error.
    int person_index = table_.shuffled_indices_for_person(i);
    core::monte_carlo::MeanVariancePair difference_mean_variance;
    this->IntegrationSampleErrorPerPerson_(
      person_index, first_sample, second_sample, &difference_mean_variance);
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

  // The following computes the CF quantity in the paper.
  double correction_factor =
    static_cast<double>(
      table_.num_people() - first_sample.num_active_people()) /
    static_cast<double>(table_.num_people() - 1);

  // Compute the average difference of simulated log probabilities.
  core::monte_carlo::MeanVariancePair average_difference;
  for(int i = 0; i < first_sample.num_active_people(); i++) {
    int person_index = table_.shuffled_indices_for_person(i);
    double difference =
      (log(first_sample.simulated_choice_probability(person_index)) -
       log(second_sample.simulated_choice_probability(person_index)));
    average_difference.push_back(difference);
  }
  return correction_factor * average_difference.sample_mean_variance();
}

template<typename TableType>
void MixedLogitDCM<TableType>::UpdateSampleAllocation_(
  const ArgumentType &arguments_in,
  double integration_sample_error,
  const SamplingType &second_sample,
  SamplingType *first_sample) const {

  std::vector<double> tmp_vector(first_sample->num_active_people());
  double total_sample_variance = 0.0;

  // Loop over each active person.
  for(int i = 0; i < first_sample->num_active_people(); i++) {

    // Get the active person index.
    int person_index = table_.shuffled_indices_for_person(i);
    core::monte_carlo::MeanVariancePair difference_mean_variance;
    IntegrationSampleErrorPerPerson_(
      person_index, *first_sample, second_sample, &difference_mean_variance);

    // Form the $S_i'$ for the current person without the
    // multiplicative factor $\sum\limits_{i \in N} s_{i, diff}$.
    tmp_vector[i] = difference_mean_variance.sample_variance() /
                    (arguments_in.gradient_norm_threshold_ *
                     integration_sample_error *
                     core::math::Sqr(second_sample.num_active_people()));
    total_sample_variance += difference_mean_variance.sample_variance();
  }

  // Loop over each active person and update.
  for(int i = 0; i < first_sample->num_active_people(); i++) {

    // Get the active person index.
    int person_index = table_.shuffled_indices_for_person(i);

    // Finalize by multiplying by the sum factor.
    tmp_vector[i] *= total_sample_variance;
    int num_additional_samples =
      std::max(
        1,
        static_cast<int>(
          ceil(
            tmp_vector[i] -
            first_sample->num_integration_samples(person_index))));

    // Add samples.
    first_sample->AddSamples(person_index, num_additional_samples);
  }
}

template<typename TableType>
void MixedLogitDCM<TableType>::Init(ArgumentType &arguments_in) {

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
  iterate->NegativeSimulatedLogLikelihoodGradient(&gradient);

  // Depending on the selected Hessian update scheme, compute it
  // differently.
  if(arguments_in.hessian_update_method_ == "exact") {

    // Exact computation.
    iterate->NegativeSimulatedLogLikelihoodHessian(&hessian);
  }
  else {

    // Approximate updates start with the identity Hessian.
    hessian.eye();
  }

  // The step direction and its 2-norm.
  arma::vec p;
  double p_norm;

  // The initial trust region radius is set to the 10 % of the maximum
  // trust region radius.
  double current_radius = 0.1 * arguments_in.max_trust_region_radius_;
  std::cerr << "The initial maximum trust region radius is set to " <<
            current_radius;

  // Enter the trust region loop.
  do {

    // Obtain the step direction by solving Equation 4.3
    // approximately.
    core::optimization::TrustRegionUtil::ObtainStepDirection(
      arguments_in.trust_region_search_method_, current_radius, gradient,
      hessian, &p, &p_norm);

    // Get the reduction ratio rho (Equation 4.4)
    SamplingType *next_iterate = new SamplingType();
    next_iterate->Init(*iterate, p);
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

      if(arguments_in.hessian_update_method_ == "exact") {
        iterate->NegativeSimulatedLogLikelihoodHessian(&hessian);
      }

      // Delete the discarded iterate.
      delete next_iterate;
      continue;
    }

    // Increase the integration sample size.
    if(sqrt(integration_sample_error) > sqrt(data_sample_error) &&
        fabs(decrease_predicted_by_model) <
        arguments_in.gradient_norm_threshold_ *
        sqrt(integration_sample_error)) {

      // Update the sample size for each active person.
      UpdateSampleAllocation_(
        arguments_in, integration_sample_error, *next_iterate, iterate);

      // Update the gradient and the hessian.
      iterate->NegativeSimulatedLogLikelihoodGradient(&gradient);

      if(arguments_in.hessian_update_method_ == "exact") {
        iterate->NegativeSimulatedLogLikelihoodHessian(&hessian);
      }

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

      // Compute the new gradient.
      arma::vec new_gradient;
      next_iterate->NegativeSimulatedLogLikelihoodGradient(&new_gradient);
      arma::mat new_hessian;
      if(arguments_in.hessian_update_method_ == "bfgs") {
        core::optimization::QuasiNewtonHessianUpdate::BFGSUpdate(
          hessian, iterate->parameters(), next_iterate->parameters(),
          gradient, new_gradient, &new_hessian);
        hessian = new_hessian;
      }
      else if(arguments_in.hessian_update_method_ == "sr1") {
        core::optimization::QuasiNewtonHessianUpdate::SymmetricRank1Update(
          hessian, iterate->parameters(), next_iterate->parameters(),
          gradient, new_gradient, &new_hessian);
        hessian = new_hessian;
      }

      // Accept the next iterate.
      delete iterate;
      iterate = next_iterate;

      // Update the gradient and the hessian.
      gradient = new_gradient;
      if(arguments_in.hessian_update_method_ == "exact") {
        iterate->NegativeSimulatedLogLikelihoodHessian(&hessian);
      }
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
    if(fabs(predicted_objective_value_improvement)  <
        arguments_in.gradient_norm_threshold_ *
        sqrt(integration_sample_error) &&
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
}
}

#endif

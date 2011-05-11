/** @file mixed_logit_dcm_dev.h
 *
 *  The implementation of mixed logit discrete choice model.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_MIXED_LOGIT_DCM_MIXED_LOGIT_DCM_DEV_H
#define MLPACK_MIXED_LOGIT_DCM_MIXED_LOGIT_DCM_DEV_H

#include "core/optimization/quasi_newton_hessian_update.h"
#include "core/optimization/trust_region_dev.h"
#include "mlpack/mixed_logit_dcm/mixed_logit_dcm.h"
#include "mlpack/mixed_logit_dcm/training_error_measure.h"

namespace mlpack {
namespace mixed_logit_dcm {

template<typename TableType, typename DistributionType>
double MixedLogitDCM<TableType, DistributionType>::GradientError_(
  const ArgumentType &arguments_in,
  const SamplingType &sample) const {

  int num_error_trials =
    (arguments_in.distribution_ == "constant") ? 1 : 30;

  // The dummy zero vector.
  arma::vec zero_vector;
  zero_vector.zeros(table_.distribution().num_parameters());

  // Accumulated samples.
  core::monte_carlo::MeanVariancePair weighted_gradient_norms;

  // Repeat until convergence.
  arma::vec gradient;
  arma::vec weighted_gradient;
  for(int j = 0; j < num_error_trials; j++) {

    // Re-sample at the two parameters with the same number of
    // samples.
    SamplingType new_sample;
    new_sample.Init(sample, zero_vector);

    // Accumulate samples.
    weighted_gradient.zeros(table_.distribution().num_parameters());
    for(int i = 0; i < sample.num_active_people(); i++) {
      int person_index = table_.shuffled_indices_for_person(i);
      sample.simulated_choice_probability_gradient(
        person_index, &gradient);
      double probability = sample.simulated_choice_probability(person_index);
      weighted_gradient += (gradient / probability);
    }
    weighted_gradient /= static_cast<double>(sample.num_active_people());
    weighted_gradient_norms.push_back(
      arma::dot(weighted_gradient, weighted_gradient));
  }

  return weighted_gradient_norms.sample_variance();
}

template<typename TableType, typename DistributionType>
double MixedLogitDCM<TableType, DistributionType>::IntegrationSampleError_(
  const ArgumentType &arguments_in,
  const SamplingType &first_sample,
  const SamplingType &second_sample,
  core::monte_carlo::MeanVariancePairVector *error_per_person) const {

  int num_error_trials =
    (arguments_in.distribution_ == "constant") ? 1 : 30;

  // The dummy zero vector.
  arma::vec zero_vector;
  zero_vector.zeros(table_.distribution().num_parameters());

  // Accumulated samples.
  error_per_person->Init(first_sample.num_active_people());

  // Repeat until convergence.
  for(int j = 0; j < num_error_trials; j++) {

    // Re-sample at the two parameters with the same number of
    // samples.
    SamplingType new_first_sample;
    new_first_sample.Init(first_sample, zero_vector);
    SamplingType new_second_sample;
    new_second_sample.Init(second_sample, zero_vector);

    // Accumulate samples.
    for(int i = 0; i < first_sample.num_active_people(); i++) {
      int person_index = table_.shuffled_indices_for_person(i);
      double first_simulated_choice_probability =
        new_first_sample.simulated_choice_probability(person_index);
      double second_simulated_choice_probability =
        new_second_sample.simulated_choice_probability(person_index);
      double difference = log(first_simulated_choice_probability) -
                          log(second_simulated_choice_probability);
      (*error_per_person)[i].push_back(difference);
    }
  }

  double integration_sample_error = 0.0;
  for(int i = 0; i < first_sample.num_active_people(); i++) {
    integration_sample_error += (*error_per_person)[i].sample_variance();
  }
  integration_sample_error /=
    core::math::Sqr(first_sample.num_active_people());
  return integration_sample_error;
}

template<typename TableType, typename DistributionType>
double MixedLogitDCM<TableType, DistributionType>::DataSampleError_(
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

template<typename TableType, typename DistributionType>
void MixedLogitDCM<TableType, DistributionType>::UpdateSampleAllocation_(
  const ArgumentType &arguments_in,
  double integration_sample_error,
  const SamplingType &second_sample,
  SamplingType *first_sample) const {

  std::vector<double> tmp_vector(first_sample->num_active_people());
  double total_sample_variance = 0.0;

  // Compute integration sample error.
  core::monte_carlo::MeanVariancePairVector
  integration_sample_error_per_person;
  IntegrationSampleError_(
    arguments_in,
    *first_sample,
    second_sample,
    &integration_sample_error_per_person);

  // Loop over each active person.
  for(int i = 0; i < first_sample->num_active_people(); i++) {

    // Form the $S_i'$ for the current person without the
    // multiplicative factor $\sum\limits_{i \in N} s_{i, diff}$.
    tmp_vector[i] = integration_sample_error_per_person[i].sample_variance() /
                    (arguments_in.gradient_norm_threshold_ *
                     integration_sample_error *
                     core::math::Sqr(second_sample.num_active_people()));
    total_sample_variance +=
      integration_sample_error_per_person[i].sample_variance();
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
    std::cerr << "  Adding " << num_additional_samples << " to Person " <<
              person_index << "\n";
    first_sample->AddSamples(person_index, num_additional_samples);
  }
}

template<typename TableType, typename DistributionType>
void MixedLogitDCM<TableType, DistributionType>::Init(
  ArgumentType &arguments_in) {

  // Initialize the table for storing/accessing the attribute vector
  // for each person.
  table_.Init(arguments_in);
}

template<typename TableType, typename DistributionType>
void MixedLogitDCM<TableType, DistributionType>::Compute(
  const ArgumentType &arguments_in,
  mlpack::mixed_logit_dcm::MixedLogitDCMResult *result_out) {

  // Here is the main entry of the algorithm.
  int initial_num_data_samples =
    std::max(
      1, static_cast<int>(
        table_.num_people() *
        arguments_in.initial_dataset_sample_rate_));
  int initial_num_integration_samples =
    std::max(
      static_cast<int>(
        arguments_in.initial_integration_sample_rate_ *
        arguments_in.max_num_integration_samples_per_person_), 36);

  // For constant distribution (multinomial logit), the number of
  // integration samples is 2. We only need 1, but let's just take 2
  // samples to prevent division by zero for sample variance.
  if(arguments_in.distribution_ == "constant") {
    initial_num_integration_samples = 2;
  }

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
    hessian.eye(
      iterate->parameters().n_elem, iterate->parameters().n_elem);
  }

  // The step direction and its 2-norm.
  arma::vec p;
  double p_norm;

  // The initial trust region radius is set to the 10 % of the maximum
  // trust region radius.
  double current_radius = 0.1 * arguments_in.max_trust_region_radius_;
  std::cerr << "The initial maximum trust region radius is set to " <<
            current_radius << "\n";

  // Enter the trust region loop.
  int num_iterations = 0;
  do {

    std::cerr << "\nThe current iterate:\n";
    iterate->parameters().print();

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

    std::cerr << "The current function value (with " <<
              iterate->num_active_people() << " people: " <<
              iterate_function_value << "; ";
    std::cerr << "The function value at the next iterate: " <<
              next_iterate_function_value << "; ";
    std::cerr << "The model reduction ratio: " << model_reduction_ratio << "\n";

    // Compute the data sample error and the integration sample error.
    double data_sample_error = this->DataSampleError_(*iterate, *next_iterate);
    core::monte_carlo::MeanVariancePairVector
    integration_sample_error_per_person;
    double integration_sample_error =
      fabs(
        this->IntegrationSampleError_(
          arguments_in,
          *iterate,
          *next_iterate,
          &integration_sample_error_per_person));

    // Determine whether the termination condition has been reached.
    if(TerminationConditionReached_(
          arguments_in, decrease_predicted_by_model,
          data_sample_error, integration_sample_error,
          *iterate, gradient, &num_iterations)) {
      delete next_iterate;
      break;
    }

    // If we are not ready to terminate yet, then consider one of the
    // three options. (1) Increase the data sample size; (2) Increase
    // the integration sample size; (3) Do a step to the new iterate.

    std::cerr << "Data sample error: " << sqrt(data_sample_error) <<
              " ; integration sample error: " <<
              sqrt(integration_sample_error) << "\n";

    // Increase the data sample size.
    if(iterate->num_active_people() < table_.num_people() &&
        sqrt(data_sample_error) >= sqrt(integration_sample_error) &&
        decrease_predicted_by_model <=
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

      std::cerr << "  Adding " << num_additional_people <<
                " additional people.\n";

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
        decrease_predicted_by_model <=
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
      std::cerr << "  Setting the trust region radius to: " <<
                current_radius << "\n";
    }
    else {
      if(model_reduction_ratio > 0.80 &&
          fabs(p_norm - current_radius) <= 0.001) {
        current_radius = 2.0 * current_radius;
        std::cerr << "  Increasing the trust region radius to: " <<
                  current_radius << "\n";
      }
    }
    const double eta = 0.05;
    if(model_reduction_ratio > eta) {

      std::cerr << "  Accepting the trust region iterate...\n";

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

      std::cerr << "  Discarding the trust region iterate...\n";

      // Delete the discarded iterate.
      delete next_iterate;
    }
  }
  while(true);

  std::cerr << "The final iterate:\n";
  iterate->parameters().print();

  // Free the iterate.
  delete iterate;
}

template<typename TableType, typename DistributionType>
bool MixedLogitDCM<TableType, DistributionType>::TerminationConditionReached_(
  const ArgumentType &arguments_in,
  double predicted_objective_value_improvement, double data_sample_error,
  double integration_sample_error, const SamplingType &sampling,
  const arma::vec &gradient,
  int *num_iterations) const {

  // Termination condition is only considered when we use all the
  // people in the sampling (outer term consists of all people).
  if(sampling.num_active_people() == table_.num_people()) {

    (*num_iterations)++;
    if(*num_iterations >= arguments_in.max_num_iterations_) {
      std::cerr << "Exceeding the maximum number of iterations of " <<
                arguments_in.max_num_iterations_ << " so terminating...\n";
      return true;
    }

    std::cerr << "  Testing the termination condition...\n";
    double predicted_objective_value_improvement_threshold =
      arguments_in.gradient_norm_threshold_ * sqrt(integration_sample_error);
    std::cerr << "    Comparing the predicted objective function improvement "
              "against the sampling error: " <<
              predicted_objective_value_improvement
              << " against " <<
              predicted_objective_value_improvement_threshold << "\n";
    std::cerr << "    Comparing the sampling error against its threshold: " <<
              sqrt(integration_sample_error) << " against " <<
              arguments_in.integration_sample_error_threshold_ << "\n";

    // If the predicted improvement in the objective value is less
    // than the integration sample error and the integration sample
    // error is small,
    if(predicted_objective_value_improvement <=
        predicted_objective_value_improvement_threshold &&
        sqrt(integration_sample_error) <=
        arguments_in.integration_sample_error_threshold_) {

      // Compute the gradient error.
      double gradient_error = this->GradientError_(arguments_in, sampling);
      double squared_gradient_error_upper_bound =
        arma::dot(gradient, gradient) +
        arguments_in.gradient_norm_threshold_ * sqrt(gradient_error);

      std::cerr << "    The upper bound on the gradient squared error: " <<
                squared_gradient_error_upper_bound << "\n";
      if(squared_gradient_error_upper_bound <= 0.001) {
        return true;
      }
    }
  }
  return false;
}
}
}

#endif

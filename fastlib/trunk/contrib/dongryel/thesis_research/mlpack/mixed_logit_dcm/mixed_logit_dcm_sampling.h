/** @file mixed_logit_dcm_sampling.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_MIXED_LOGIT_DCM_MIXED_LOGIT_DCM_SAMPLING_H
#define MLPACK_MIXED_LOGIT_DCM_MIXED_LOGIT_DCM_SAMPLING_H

#include <armadillo>
#include <vector>
#include "core/monte_carlo/mean_variance_pair_matrix.h"

namespace mlpack {
namespace mixed_logit_dcm {
template<typename DCMTableType>
class MixedLogitDCMSampling {
  private:

    /** @brief The simulated choice probabilities (sample mean
     *         and sample variance information).
     */
    std::vector< core::monte_carlo::MeanVariancePair >
    simulated_choice_probabilities_;

    /** @brief The gradient of the simulated log likelihood per person.
     */
    std::vector <
    core::monte_carlo::MeanVariancePairVector >
    simulated_loglikelihood_gradients_;

    /** @brief The Hessian of the simulated log likelihood per
     *         person. Each is composed of a pair, the first of which
     *         is, $\frac{\partial}{\partial \theta}
     *         \beta^{\nu}(\theta) \frac{\partial^2}{\partial \beta^2}
     *         P_{i,j_i^*}(\beta^{nu}(\theta))
     *         (\frac{\partial}{\partial \theta}
     *         \beta^{\nu}(\theta))^T$. The second is $\frac{\partial}
     *         {\partial \theta} \beta^{\nu}(\theta) \frac{\partial}
     *         {\beta} {P_{i,j_i^*} (\beta^{\nu}(\theta))$. This vector
     *         keeps track of Equation 8.14.
     */
    std::vector <
    std::pair < core::monte_carlo::MeanVariancePairMatrix,
        core::monte_carlo::MeanVariancePairVector > >
        simulated_loglikelihood_hessians_;

    /** @brief The pointer to the discrete choice model table from
     *         which we access the attribute information of different
     *         discrete choices.
     */
    DCMTableType *dcm_table_;

    /** @brief The number of active outer-terms in the simulated
     *         log-likelihood score.
     */
    int num_active_people_;

    std::vector<int> num_integration_samples_;

  private:

    /** @brief Computes the choice probability vector for the
     *         person_index-th person for each of his/her potential
     *         choices given the vector $\beta$. This is $P_{i,j}$ in
     *         a long vector form.
     */
    void ComputeChoiceProbabilities_(
      int person_index, const arma::vec &beta_vector,
      arma::vec *choice_probabilities) {

      int num_discrete_choices = dcm_table_->num_discrete_choices(person_index);
      choice_probabilities->set_size(num_discrete_choices);

      // First compute the normalizing sum.
      double normalizing_sum = 0.0;
      for(int discrete_choice_index = 0;
          discrete_choice_index < num_discrete_choices;
          discrete_choice_index++) {

        // Grab each attribute vector and take a dot product between
        // it and the beta vector.
        arma::vec attribute_for_discrete_choice;
        dcm_table_->get_attribute_vector(
          person_index, discrete_choice_index, &attribute_for_discrete_choice);
        double dot_product =
          arma::dot(
            beta_vector, attribute_for_discrete_choice);
        double unnormalized_probability = exp(dot_product);
        normalizing_sum += unnormalized_probability;
        (*choice_probabilities)[discrete_choice_index] =
          unnormalized_probability;
      }

      // Then, normalize.
      for(int discrete_choice_index = 0;
          discrete_choice_index < num_discrete_choices;
          discrete_choice_index++) {
        (*choice_probabilities)[discrete_choice_index] /= normalizing_sum;
      }
    }

  public:

    double simulated_choice_probability(int person_index) const {
      return simulated_choice_probabilities_[person_index].sample_mean();
    }

    /** @brief Returns the Hessian of the current simulated log
     *         likelihood score objective. This completes the
     *         computation of Equation 8.14 in the paper.
     */
    void SimulatedLoglikelihoodHessian(
      arma::mat *likelihood_hessian) const {

      likelihood_hessian->set_size(
        dcm_table_->num_parameters(), dcm_table_->num_parameters());
      likelihood_hessian->zeros();

      // For each active person,
      for(int i = 0; i < num_active_people_; i++) {

        // Get the index in the shuffled indices to find out the ID of
        // the person in the sample pool.
        int person_index = dcm_table_->shuffled_indices_for_person(i);

        // Get the simulated choice probability for the given person.
        int discrete_choice_index =
          this->get_discrete_choice_index(person_index);
        double simulated_choice_probability =
          this->simulated_choice_probability(person_index);
        double inverse_simulated_choice_probability =
          1.0 / simulated_choice_probability;

        // Get the components for the Hessian for the person.
        const core::monte_carlo::MeanVariancePairMatrix &hessian_first_part =
          simulated_loglikelihood_hessians_[person_index].first;
        const core::monte_carlo::MeanVariancePairVector &hessian_second_part =
          simulated_loglikelihood_hessians_[person_index].second;
        arma::mat hessian_first;
        arma::vec hessian_second;
        hessian_first_part.sample_means(&hessian_first);
        hessian_second_part.sample_means(&hessian_second);

        // Construct the contribution on the fly.
        (*likelihood_hessian) += inverse_simulated_choice_probability *
                                 hessian_first;
        (*likelihood_hessian) +=
          (- core::math::Sqr(inverse_simulated_choice_probability)) *
          hessian_second * arma::trans(hessian_second);
      }

      // Divide by the number of people.
      (*likelihood_hessian) =
        (1.0 / static_cast<double>(num_active_people_)) *
        (*likelihood_hessian);
    }

    /** @brief Return the gradient of the current simulated log
     *         likelihood score objective. This computes Equation 8.7
     *         in the paper.
     */
    void SimulatedLoglikelihoodGradient(
      arma::vec *likelihood_gradient) const {

      likelihood_gradient->set_size(dcm_table_->num_parameters());
      likelihood_gradient->zeros();

      // For each active person,
      for(int i = 0; i < num_active_people_; i++) {

        // Get the index in the shuffled indices to find out the ID of
        // the person in the sample pool.
        int person_index = dcm_table_->shuffled_indices_for_person(i);

        // Get the simulated choice probability for the given person.
        double simulated_choice_probability =
          this->simulated_choice_probability(person_index);
        double inverse_simulated_choice_probability =
          1.0 / simulated_choice_probability;

        // Get the gradient for the person.
        const core::monte_carlo::MeanVariancePairVector &gradient =
          this->simulated_loglikelihood_gradient(person_index);
        arma::vec gradient_vector;
        gradient.sample_means(&gradient_vector);

        // Add the inverse probability weighted gradient vector for
        // the current person to the total tally.
        (*likelihood_gradient) =
          (*likelihood_gradient) +
          inverse_simulated_choice_probability * gradient_vector;
      }

      // Divide by the number of people.
      (*likelihood_gradient) =
        (1.0 / static_cast<double>(num_active_people_)) *
        (*likelihood_gradient);
    }

    /** @brief Return the current simulated log likelihood score.
     */
    double SimulatedLogLikelihood() const {
      double current_simulated_log_likelihood = 0;
      for(int i = 0; i < num_active_people_; i++) {

        // Get the index in the shuffled indices to find out the ID of
        // the person in the sample pool.
        int person_index = dcm_table_->shuffled_indices_for_person(i);

        // Get the simulated choice probability for the
        // given person corresponding to its discrete choice.
        double simulated_choice_probability =
          this->simulated_choice_probability(person_index);
        current_simulated_log_likelihood +=
          log(simulated_choice_probability);
      }
      current_simulated_log_likelihood /=
        static_cast<double>(num_active_people_);
      return current_simulated_log_likelihood;
    }

    /** @brief Adds an integration sample to the person_index-th
     *         person so that the person's running simulated choice
     *         probabilities can be updated.
     */
    void AddIntegrationSample(
      int person_index, const arma::vec &beta_vector) {

      // Given the parameter vector, compute the choice probabilities.
      arma::vec choice_probabilities;
      ComputeChoiceProbabilities_(
        person_index, beta_vector, &choice_probabilities);

      // Given the beta vector, compute the products between the
      // gradient of the $\beta$ with respect to $\theta$ and
      // $\bar{X}_i res_{i,j_i^*}(\beta^v(\theta))$.
      arma::vec beta_gradient_product;

      // j_i^* index.
      int discrete_choice_index =
        dcm_table_->get_discrete_choice_index(person_index);
      dcm_table_->distribution()->MixedLogitParameterGradientProducts(
        this, person_index, discrete_choice_index, beta_vector,
        choice_probabilities, &beta_gradient_product);

      // Update the simulated choice probabilities
      // and the simulated log-likelihood gradients.
      simulated_choice_probabilities_[person_index].push_back(
        choice_probabilities[discrete_choice_index]);

      // Simulated log-likelihood gradient update by the simulated
      // choice probabilty scaled gradient product.
      beta_gradient_product = choice_probabilities[discrete_choice_index] *
                              beta_gradient_product;
      simulated_loglikelihood_gradients_[person_index].push_back(
        beta_gradient_product);

      // Update the Hessian of the simulated loglikelihood for the
      // given person.
      arma::mat hessian_first_part;
      arma::vec hessian_second_part;
      dcm_table_->distribution()->HessianProducts(
        this, person_index, discrete_choice_index, beta_vector,
        choice_probabilities, &hessian_first_part, &hessian_second_part);
      simulated_loglikelihood_hessians_[person_index].first.push_back(
        hessian_first_part);
      simulated_loglikelihood_hessians_[person_index].second.push_back(
        hessian_second_part);
    }

    void Init(
      DCMTableType *dcm_table_in,
      int num_active_people_in,
      const std::vector<int> &num_integration_samples_in) {

      dcm_table_ = dcm_table_in;
      num_active_people_ = num_active_people_in;
      num_integration_samples_ = num_integration_samples_in;

      // This vector maintains the running simulated choice
      // probabilities per person.
      simulated_choice_probabilities_.resize(dcm_table_->num_people());

      // This vector maintains the gradients of the simulated
      // loglikelihood per person per discrete choice.
      simulated_loglikelihood_gradients_.resize(dcm_table_->num_people());
      for(unsigned int i = 0; i < simulated_loglikelihood_gradients_.size();
          i++) {
        simulated_loglikelihood_gradients_[i].Init(
          dcm_table_->num_parameters());
      }

      // This vector maintains the Hessians of the simulated
      // loglikelihood per person per discrete choice.
      simulated_loglikelihood_hessians_.resize(dcm_table_->num_people());

      // Build up the samples.

    }

    void AddActivePeople(int num_additional_people) {
      num_active_people_ += num_additional_people;

      // Build up additional samples for the new people.
    }

    const std::vector <
    core::monte_carlo::MeanVariancePair > &simulated_choice_probabilities() {
      return simulated_choice_probabilities_;
    }

    const std::vector <
    core::monte_carlo::MeanVariancePairVector > &
    simulated_loglikelihood_gradients() {
      return simulated_loglikelihood_gradients_;
    }

    const std::vector <
    std::pair < core::monte_carlo::MeanVariancePairMatrix,
        core::monte_carlo::MeanVariancePairVector > > &
    simulated_loglikelihood_hessians() {
      return simulated_loglikelihood_hessians_;
    }
};
};
};

#endif

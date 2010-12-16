/** @file dcm_table.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_MIXED_LOGIT_DCM_DCM_TABLE_H
#define MLPACK_MIXED_LOGIT_DCM_DCM_TABLE_H

#include <algorithm>
#include <vector>
#include "core/table/table.h"
#include "core/math/linear_algebra.h"
#include "core/monte_carlo/mean_variance_pair.h"
#include "core/monte_carlo/mean_variance_pair_matrix.h"
#include "mlpack/mixed_logit_dcm/mixed_logit_dcm_distribution.h"

namespace mlpack {
namespace mixed_logit_dcm {
template<typename TableType>
class DCMTable {
  public:

    typedef DCMTable<TableType> DCMTableType;

  private:

    /** @brief The distribution from which each $\beta$ is sampled
     *         from.
     */
    mlpack::mixed_logit_dcm::MixedLogitDCMDistribution <
    DCMTableType > *distribution_;

    /** @brief The pointer to the attribute vector for each person per
     *         his/her discrete choice.
     */
    TableType *attribute_table_;

    /** @brief The index of the discrete choice and the number of
     *         discrete choices made by each person (in a
     *         column-oriented matrix table form).
     */
    TableType *discrete_choice_set_info_;

    /** @brief The cumulative distribution on the number of discrete
     *         choices on the person scale.
     */
    std::vector<int> cumulative_num_discrete_choices_;

    /** @brief Used for sampling the outer-term of the simulated
     *         log-likelihood score.
     */
    std::vector<int> shuffled_indices_for_person_;

    /** @brief The number of active outer-terms in the simulated
     *         log-likelihood score.
     */
    int num_active_people_;

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

    /** @brief The Hessian of the simulated log likelihood per person
     */
    std::vector <
    core::monte_carlo::MeanVariancePairMatrix >
    simulated_loglikelihood_hessians_;

  private:

    /** @brief Computes the choice probability vector for the person_index-th
     *         person for each of his/her potential choices given the
     *         parameter vector $\beta$. This is $P_{i,j}$ in a long vector
     *         form.
     */
    void ComputeChoiceProbabilities_(
      int person_index, const core::table::DensePoint &parameter_vector,
      core::table::DensePoint *choice_probabilities) {

      int num_discrete_choices = this->num_discrete_choices(person_index);
      choice_probabilities->Init(num_discrete_choices);

      // First compute the normalizing sum.
      double normalizing_sum = 0.0;
      for(int discrete_choice_index = 0;
          discrete_choice_index < num_discrete_choices;
          discrete_choice_index++) {

        // Grab each attribute vector and take a dot product between
        // it and the parameter vector.
        core::table::DensePoint attribute_for_discrete_choice;
        this->get_attribute_vector(
          person_index, discrete_choice_index, &attribute_for_discrete_choice);
        double dot_product = core::math::Dot(
                               parameter_vector, attribute_for_discrete_choice);
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

    /** @brief Returns the number of discrete choices available for
     *         the given person.
     */
    int num_discrete_choices(int person_index) const {
      return static_cast<int>(
               discrete_choice_set_info_->data().get(1, person_index));
    }

    int get_discrete_choice_index(int person_index) const {
      return static_cast<int>(
               discrete_choice_set_info_->data().get(0, person_index));
    }

    /** @brief Return the gradient of the current simulated log
     *         likelihood score objective. This computes Equation 8.7
     *         in the paper.
     */
    void SimulatedLoglikelihoodGradient(
      core::table::DensePoint *likelihood_gradient) const {

      likelihood_gradient->Init(distribution_->num_parameters());
      likelihood_gradient->SetZero();

      // For each active person,
      for(int i = 0; i < num_active_people_; i++) {

        // Get the index in the shuffled indices to find out the ID of
        // the person in the sample pool.
        int person_index = shuffled_indices_for_person_[i];

        // Get the simulated choice probability for the given person.
        int discrete_choice_index =
          this->get_discrete_choice_index(person_index);
        double simulated_choice_probability =
          this->simulated_choice_probability(person_index);
        double inverse_simulated_choice_probability =
          1.0 / simulated_choice_probability;

        // Get the gradient for the person.
        const core::monte_carlo::MeanVariancePairVector &gradient =
          this->simulated_loglikelihood_gradient(person_index);
        core::table::DensePoint gradient_vector;
        gradient.sample_means(&gradient_vector);

        // Add the inverse probability weighted gradient vector for
        // the current person to the total tally.
        core::math::AddExpert(
          inverse_simulated_choice_probability,
          gradient_vector, likelihood_gradient);
      }

      // Divide by the number of people.
      core::math::Scale(
        1.0 / static_cast<double>(num_active_people_), likelihood_gradient);
    }

    /** @brief Return the current simulated log likelihood score.
     */
    double SimulatedLogLikelihood() const {
      double current_simulated_log_likelihood = 0;
      for(int i = 0; i < num_active_people_; i++) {

        // Get the index in the shuffled indices to find out the ID of
        // the person in the sample pool.
        int person_index = shuffled_indices_for_person_[i];

        // Get the highest simulated choice probability for the
        // given person.
        int discrete_choice_index =
          this->get_discrete_choice_index(person_index);
        double simulated_choice_probability =
          this->simulated_choice_probability(person_index);
        current_simulated_log_likelihood +=
          log(simulated_choice_probability);
      }
      current_simulated_log_likelihood /=
        static_cast<double>(num_active_people_);
      return current_simulated_log_likelihood;
    }

    int num_people() const {
      return static_cast<int>(cumulative_num_discrete_choices_.size());
    }

    template<typename ArgumentType>
    void Init(ArgumentType &argument_in) {

      // Set the incoming attributes table and the number of choices
      // per person in the list.
      attribute_table_ = argument_in.attribute_table_;
      discrete_choice_set_info_ = argument_in.num_discrete_choices_per_person_;

      // Initialize a randomly shuffled vector of indices for sampling
      // the outer term in the simulated log-likelihood.
      shuffled_indices_for_person_.resize(
        argument_in.num_discrete_choices_per_person_->n_entries());
      for(unsigned int i = 0; i < shuffled_indices_for_person_.size(); i++) {
        shuffled_indices_for_person_[i] = i;
      }
      std::random_shuffle(
        shuffled_indices_for_person_.begin(),
        shuffled_indices_for_person_.end());
      num_active_people_ = 0;

      // Compute the cumulative distribution on the number of
      // discrete choices so that we can return the right column
      // index in the attribute table for given (person, discrete
      // choice) pair.
      cumulative_num_discrete_choices_.resize(
        argument_in.num_discrete_choices_per_person_->n_entries());
      cumulative_num_discrete_choices_[0] = 0;
      for(unsigned int i = 1; i < cumulative_num_discrete_choices_.size();
          i++) {
        core::table::DensePoint point;
        argument_in.num_discrete_choices_per_person_->get(i - 1, &point);
        int num_choices_for_current_person = static_cast<int>(point[1]);
        cumulative_num_discrete_choices_[i] =
          cumulative_num_discrete_choices_[i - 1] +
          num_choices_for_current_person;
      }

      // Do a quick check to make sure that the cumulative
      // distribution on the number of choices match up the total
      // number of attribute vectors. Otherwise, quit.
      core::table::DensePoint last_count_vector;
      argument_in.num_discrete_choices_per_person_->get(
        cumulative_num_discrete_choices_.size() - 1, &last_count_vector);
      int last_count = static_cast<int>(last_count_vector[0]);
      if(cumulative_num_discrete_choices_[
            cumulative_num_discrete_choices_.size() - 1] +
          last_count != argument_in.attribute_table_->n_entries()) {
        std::cerr << "The total number of discrete choices do not equal "
                  "the number of total number of attribute vectors.\n";
        exit(0);
      }

      // This vector maintains the running simulated choice
      // probabilities per person.
      simulated_choice_probabilities_.resize(
        shuffled_indices_for_person_.size());

      // This vector maintains the gradients of the simulated
      // loglikelihood per person per discrete choice.
      simulated_loglikelihood_gradients_.resize(
        shuffled_indices_for_person_.size());
      for(unsigned int i = 0; i < simulated_loglikelihood_gradients_.size();
          i++) {
        simulated_loglikelihood_gradients_[i].Init(
          distribution_->num_parameters());
      }

      // This vector maintains the Hessians of the simulated
      // loglikelihood per person per discrete choice.
      simulated_loglikelihood_hessians_.resize(
        shuffled_indices_for_person_.size());
    }

    /** @brief Adds the specified number of terms to the outer sum,
     *         where each term corresponds to a person.
     */
    void AddPeople(int num_people) {
      num_active_people_ += num_people;
    }

    /** @brief Adds an integration sample to the person_index-th
     *         person so that the person's running simulated choice
     *         probabilities can be updated.
     */
    void AddIntegrationSample(
      int person_index, const core::table::DensePoint &parameter_vector) {

      // Given the parameter vector, compute the choice probabilities.
      core::table::DensePoint choice_probabilities;
      ComputeChoiceProbabilities_(
        person_index, parameter_vector, &choice_probabilities);

      // Given the parameter vector, compute the products between the
      // gradient of the $\beta$ with respect to $\theta$ and
      // $\bar{X}_i res_{i,j_i^*}(\beta^v(\theta))$.
      core::table::DensePoint beta_gradient_products;

      // j_i^* index.
      int discrete_choice_index = this->get_discrete_choice_index(person_index);
      distribution_->MixedLogitParameterGradientProducts(
        attribute_table_, person_index,
        discrete_choice_index, parameter_vector,
        choice_probabilities, &beta_gradient_products);

      // Update the simulated choice probabilities
      // and the simulated log-likelihood gradients.
      simulated_choice_probabilities_[person_index].push_back(
        choice_probabilities[discrete_choice_index]);

      // Simulated log-likelihood gradient update.


      // simulated_loglikelihood_gradients_[index].push_back();

    }

    const core::monte_carlo::MeanVariancePairVector &
    simulated_loglikelihood_gradient(int person_index) const {
      return simulated_loglikelihood_gradients_[person_index];
    }

    double simulated_choice_probability(int person_index) const {
      return simulated_choice_probabilities_[person_index].sample_mean();
    }

    /** @brief Retrieve the discrete_choice_index-th attribute vector
     *         for the person person_index.
     */
    void get_attribute_vector(
      int person_index, int discrete_choice_index,
      core::table::DensePoint *attribute_for_discrete_choice_out) {

      int index = cumulative_num_discrete_choices_[person_index] +
                  discrete_choice_index;
      attribute_table_->get(index, attribute_for_discrete_choice_out);
    }
};
};
};

#endif

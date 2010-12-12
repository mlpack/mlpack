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

namespace mlpack {
namespace mixed_logit_dcm {
template<typename TableType>
class DCMTable {
  private:

    TableType *attribute_table_;

    std::vector<int> shuffled_indices_for_person_;

    int num_active_people_;

    std::vector< core::monte_carlo::MeanVariancePair >
    simulated_choice_probabilities_;

    std::vector<int> cumulative_num_discrete_choices_;

    std::vector<int> num_discrete_choices_per_person_;

  private:

    /** @brief Computes the choice probability vector for the person_index-th
     *         person for each of his/her potential choices given the
     *         parameter vector $\beta$. This is $P_{i,j}$ in a long vector
     *         form.
     */
    void ComputeChoiceProbabilities_(
      int person_index, const core::table::DensePoint &parameter_vector,
      core::table::DensePoint *choice_probabilities) {

      choice_probabilities->Init(
        num_discrete_choices_per_person_[person_index]);

      // First compute the normalizing sum.
      double normalizing_sum = 0.0;
      for(int discrete_choice_index = 0; discrete_choice_index <
          num_discrete_choices_per_person_[person_index];
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
      for(int discrete_choice_index = 0; discrete_choice_index <
          num_discrete_choices_per_person_[person_index];
          discrete_choice_index++) {
        (*choice_probabilities)[discrete_choice_index] /= normalizing_sum;
      }
    }

  public:

    /** @brief Output the current simulated log likelihood score.
     */
    double simulated_log_likelihood() const {
      double current_simulated_log_likelihood = 0;
      for(int i = 0; i < num_active_people_; i++) {

        // Get the index in the shuffled indices to find out the ID of
        // the person in the sample pool.
        int person_index = shuffled_indices_[i];

        // Examine the simulated choice probabilities for the given
        // person, and select the discrete choice with the highest
        // probability.
      }
      return current_simulated_log_likelihood;
    }

    int num_people() const {
      return static_cast<int>(cumulative_num_discrete_choices_.size());
    }

    void Init(
      TableType *attribute_table_in,
      TableType *num_discrete_choices_per_person_in) {

      // Set the incoming attributes table and the number of choices
      // per person in the list.
      attribute_table_ = attribute_table_in;
      num_discrete_choices_per_person_.resize(
        num_discrete_choices_per_person_in->n_entries());

      // This vector maintains the running simulated choice
      // probabilities per person per discrete choice. It is indexed
      // in the same way as the attribute table.
      simulated_choice_probabilities_.resize(attribute_table_->n_entries());

      // Initialize a randomly shuffled vector of indices for sampling
      // the outer term in the simulated log-likelihood.
      shuffled_indices_.resize(
        num_discrete_choices_per_person_in->n_entries());
      for(unsigned int i = 0; i < shuffled_indices_.size(); i++) {
        shuffled_indices_[i] = i;
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
        num_discrete_choices_per_person_in->n_entries());
      cumulative_num_discrete_choices_[0] = 0;
      for(unsigned int i = 1; i < cumulative_num_discrete_choices_.size();
          i++) {
        core::table::DensePoint point;
        num_discrete_choices_per_person_in->get(i - 1, &point);
        int num_choices_for_current_person =
          static_cast<int>(point[0]);
        cumulative_num_discrete_choices_[i] =
          cumulative_num_discrete_choices_[i - 1] +
          num_choices_for_current_person;
        num_discrete_choices_per_person_[i - 1] =
          num_choices_for_current_person;
      }

      // Do a quick check to make sure that the cumulative
      // distribution on the number of choices match up the total
      // number of attribute vectors. Otherwise, quit.
      core::table::DensePoint last_count_vector;
      num_discrete_choices_per_person_in->get(
        cumulative_num_discrete_choices_.size() - 1, &last_count_vector);
      num_discrete_choices_per_person_[
        cumulative_num_discrete_choices_.size() - 1 ] = last_count_vector[0];
      int last_count = static_cast<int>(last_count_vector[0]);
      if(cumulative_num_discrete_choices_[
            cumulative_num_discrete_choices_.size() - 1] +
          last_count != attribute_table_in->n_entries()) {
        std::cerr << "The total number of discrete choices do not equal "
                  "the number of total number of attribute vectors.\n";
        exit(0);
      }
    }

    /** @brief Adds an integration sample to the person_index-th
     *         person so that the person's running simulated choice
     *         probabilities can be updated.
     */
    void add_integration_sample(
      int person_index, const core::table::DensePoint &parameter_vector) {

      // Given the parameter vector, compute the choice probabilities.
      core::table::DensePoint choice_probabilities;
      ComputeChoiceProbabilities_(
        person_index, parameter_vector, &choice_probabilities);

      int index = cumulative_num_discrete_choices_[person_index];
      for(int num_discrete_choices = 0; num_discrete_choices <
          num_discrete_choices_per_person_[person_index];
          num_discrete_choices++, index++) {

        simulated_choice_probabilities_[index].push_back(
          choice_probabilities[num_discrete_choices]);
      }
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

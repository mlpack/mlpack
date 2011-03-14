/** @file dcm_table.h
 *
 *  The header file that maintains the discrete choice information
 *  with the distribution information.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_MIXED_LOGIT_DCM_DCM_TABLE_H
#define MLPACK_MIXED_LOGIT_DCM_DCM_TABLE_H

#include <armadillo>
#include <algorithm>
#include <vector>
#include "core/table/table.h"
#include "core/monte_carlo/mean_variance_pair.h"
#include "core/monte_carlo/mean_variance_pair_matrix.h"
#include "mlpack/mixed_logit_dcm/distribution.h"
#include "mlpack/mixed_logit_dcm/mixed_logit_dcm_sampling.h"

namespace mlpack {
namespace mixed_logit_dcm {

/** @brief A table that basically maintains the discrete choice for
 *         each person.
 */
template<typename TableType, typename DistributionType>
class DCMTable {
  public:

    typedef DCMTable<TableType, DistributionType> DCMTableType;

  private:

    /** @brief The distribution from which each $\beta$ is sampled
     *         from.
     */
    mlpack::mixed_logit_dcm::Distribution<DistributionType> distribution_;

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

  public:

    /** @brief Computes the choice probability vector for the
     *         person_index-th person for each of his/her potential
     *         choices given the vector $\beta$. This is $P_{i,j}$ in
     *         a long vector form.
     */
    void choice_probabilities(
      int person_index, const arma::vec &beta_vector,
      arma::vec *choice_probabilities) const {

      int num_discrete_choices = this->num_discrete_choices(person_index);
      choice_probabilities->set_size(num_discrete_choices);

      // First compute the normalizing sum.
      double normalizing_sum = 0.0;

      // The maximum dot product.
      double max_dot_product = - std::numeric_limits<double>::max();

      for(int discrete_choice_index = 0;
          discrete_choice_index < num_discrete_choices;
          discrete_choice_index++) {

        // Grab each attribute vector and take a dot product between
        // it and the beta vector.
        arma::vec attribute_for_discrete_choice;
        this->get_attribute_vector(
          person_index, discrete_choice_index, &attribute_for_discrete_choice);
        double dot_product =
          arma::dot(
            beta_vector, attribute_for_discrete_choice);

        // We need to watch out for a potential numerical instability
        // here. There, we just store the dot products here.
        (*choice_probabilities)[discrete_choice_index] = dot_product;
        max_dot_product = std::max(max_dot_product, dot_product);
      }

      // Compute the normalizing sum by looking at the distribution of
      // the dot-products. We want to keep each dot product to be
      // between 0 and 1 to avoid numerical overflow/underflows.
      for(int discrete_choice_index = 0;
          discrete_choice_index < num_discrete_choices;
          discrete_choice_index++) {

        double adjusted_dot_product =
          (*choice_probabilities)[discrete_choice_index] - max_dot_product + 1;
        double unnormalized_probability = exp(adjusted_dot_product);
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

    /** @brief Computes the choice probability for the given person
     *         for his/her discrete choice for a given realization of
     *         $\beta$.
     */
    double choice_probability(
      int person_index, const arma::vec &beta_vector) const {
      arma::vec choice_probabilities;
      this->choice_probabilities(
        person_index, beta_vector, &choice_probabilities);
      return choice_probabilities[
               this->get_discrete_choice_index(person_index)];
    }

    /** @brief Returns the distribution from which each $\beta$ is
     *         sampled.
     */
    const mlpack::mixed_logit_dcm::Distribution <
    DistributionType > &distribution() const {
      return distribution_;
    }

    /** @brief Returns the real person index for the $pos$-th person
     *         in the shuffled list.
     */
    int shuffled_indices_for_person(int pos) const {
      return shuffled_indices_for_person_[pos];
    }

    /** @brief Returns the number of attributes for a given discrete
     *         choice.
     */
    int num_attributes() const {
      return attribute_table_->n_attributes();
    }

    /** @brief Returns the number of discrete choices available for
     *         the given person.
     */
    int num_discrete_choices(int person_index) const {
      return static_cast<int>(
               discrete_choice_set_info_->data().get(1, person_index));
    }

    /** @brief Returns the discrete choice index for the given person.
     */
    int get_discrete_choice_index(int person_index) const {
      return static_cast<int>(
               discrete_choice_set_info_->data().get(0, person_index));
    }

    /** @brief Returns the number of parameters.
     */
    int num_parameters() const {
      return distribution_.num_parameters();
    }

    /** @brief Returns the total number of people.
     */
    int num_people() const {
      return static_cast<int>(cumulative_num_discrete_choices_.size());
    }

    /** @brief Initializes the discrete choice model table.
     */
    template<typename ArgumentType>
    void Init(ArgumentType &argument_in) {

      // Set the incoming attributes table and the number of choices
      // per person in the list.
      attribute_table_ = argument_in.attribute_table_;
      discrete_choice_set_info_ = argument_in.discrete_choice_set_info_;

      // Initialize the distribution.
      distribution_.Init(attribute_table_->n_attributes());

      // Initialize a randomly shuffled vector of indices for sampling
      // the outer term in the simulated log-likelihood.
      shuffled_indices_for_person_.resize(
        argument_in.discrete_choice_set_info_->n_entries());
      for(unsigned int i = 0; i < shuffled_indices_for_person_.size(); i++) {
        shuffled_indices_for_person_[i] = i;
      }
      std::random_shuffle(
        shuffled_indices_for_person_.begin(),
        shuffled_indices_for_person_.end());

      // Compute the cumulative distribution on the number of
      // discrete choices so that we can return the right column
      // index in the attribute table for given (person, discrete
      // choice) pair.
      cumulative_num_discrete_choices_.resize(
        argument_in.discrete_choice_set_info_->n_entries());
      cumulative_num_discrete_choices_[0] = 0;
      for(unsigned int i = 1; i < cumulative_num_discrete_choices_.size();
          i++) {
        arma::vec point;
        argument_in.discrete_choice_set_info_->get(i - 1, &point);
        int num_choices_for_current_person = static_cast<int>(point[1]);
        cumulative_num_discrete_choices_[i] =
          cumulative_num_discrete_choices_[i - 1] +
          num_choices_for_current_person;
      }

      // Do a quick check to make sure that the cumulative
      // distribution on the number of choices match up the total
      // number of attribute vectors. Otherwise, quit.
      arma::vec last_count_vector;
      argument_in.discrete_choice_set_info_->get(
        cumulative_num_discrete_choices_.size() - 1, &last_count_vector);
      int last_count = static_cast<int>(last_count_vector[1]);
      if(cumulative_num_discrete_choices_[
            cumulative_num_discrete_choices_.size() - 1] +
          last_count != argument_in.attribute_table_->n_entries()) {
        std::cerr << "The total number of discrete choices do not equal "
                  "the number of total number of attribute vectors.\n";
        exit(0);
      }
    }

    /** @brief Retrieve the discrete_choice_index-th attribute vector
     *         for the person person_index.
     */
    void get_attribute_vector(
      int person_index, int discrete_choice_index,
      arma::vec *attribute_for_discrete_choice_out) const {

      int index = cumulative_num_discrete_choices_[person_index] +
                  discrete_choice_index;
      attribute_table_->get(index, attribute_for_discrete_choice_out);
    }
};
}
}

#endif

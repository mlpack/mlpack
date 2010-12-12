/** @file dcm_table.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_MIXED_LOGIT_DCM_DCM_TABLE_H
#define MLPACK_MIXED_LOGIT_DCM_DCM_TABLE_H

#include <vector>
#include "core/table/table.h"
#include "core/math/linear_algebra.h"

namespace mlpack {
namespace mixed_logit_dcm {
template<typename TableType>
class DCMTable {
  private:

    TableType *attribute_table_;

    std::vector<int> cumulative_num_discrete_choices_;

    std::vector<int> num_discrete_choices_per_person_;

  public:

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

    /** @brief Computes the simulated choice probability for the
     *         person_index-th person, given that the person chooses
     *         the discrete_choice_index-th item. This is SP_{i, j_i^*}^{R_i}.
     */
    double simulated_choice_probability(
      int person_index, int discrete_choice_index) {


    }

    /** @brief Computes the choice probability vector for the person_index-th
     *         person for each of his/her potential choices given the
     *         parameter vector $\beta$. This is $P_{i,j}$ in a long vector
     *         form.
     */
    void choice_probabilities(
      int person_index, const core::table::DensePoint &parameter_vector,
      core::table::DensePoint *choice_probabilities) {

      choice_probabilities->Init(
        num_discrete_choices_per_person_[person_index]);

      // First compute the normalizing sum.
      int index = cumulative_num_discrete_choices_[person_index];
      double normalizing_sum = 0.0;
      for(int num_discrete_choices = 0; num_discrete_choices <
          num_discrete_choices_per_person_[person_index];
          num_discrete_choices++, index++) {

        // Grab each attribute vector and take a dot product between
        // it and the parameter vector.
        core::table::DensePoint attribute_for_discrete_choice;
        attribute_table_->get(index, &attriute_for_discrete_choice);
        double dot_product = core::math::Dot(
                               parameter_vector, attribute_for_discrete_choice);
        double unnormalized_probability = exp(dot_product);
        normalizing_sum += unnormalized_probability;
        (*choice_probabilities)[num_discrete_choices] =
          unnormalized_probability;
      }

      // Then, normalize.
      for(int num_discrete_choices = 0; num_discrete_choices <
          num_discrete_choices_per_person_[person_index];
          num_discrete_choices++) {
        (*choice_probabilities)[num_discrete_choices] /= normalizing_sum;
      }
    }

    /** @brief Retrieve the discrete_choice_index-th attribute vector
     *         for the person person_index.
     */
    void get(
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

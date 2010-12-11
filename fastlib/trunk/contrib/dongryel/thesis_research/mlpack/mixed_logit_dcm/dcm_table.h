/** @file dcm_table.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_MIXED_LOGIT_DCM_DCM_TABLE_H
#define MLPACK_MIXED_LOGIT_DCM_DCM_TABLE_H

#include <vector>
#include "core/table/table.h"

namespace mlpack {
namespace mixed_logit_dcm {
template<typename TableType>
class DCMTable {
  private:

    TableType *attribute_table_;

    std::vector<double> cumulative_num_discrete_choices_;

  public:

    int num_people() const {
      return static_cast<int>(cumulative_num_discrete_choices_.size());
    }

    void Init(
      TableType *attribute_table_in,
      TableType *num_discrete_choices_per_person_in) {

      // Set the incoming attributes table.
      attribute_table_ = attribute_table_in;

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
      }

      // Do a quick check to make sure that the cumulative
      // distribution on the number of choices match up the total
      // number of attribute vectors. Otherwise, quit.
      core::table::DensePoint last_count_vector;
      num_discrete_choices_per_person_in->get(
        cumulative_num_discrete_choices_.size() - 1, &last_count_vector);
      int last_count = static_cast<int>(last_count_vector[0]);
      if(cumulative_num_discrete_choices_[
            cumulative_num_discrete_choices_.size() - 1] +
          last_count != attribute_table_in->n_entries()) {
        std::cerr << "The total number of discrete choices do not equal "
                  "the number of total number of attribute vectors.\n";
        exit(0);
      }
    }

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

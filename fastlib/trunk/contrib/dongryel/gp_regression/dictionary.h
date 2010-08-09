/** @file dictionary.h
 *
 *  @brief A generic dictionary for subset of regressor methods.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef ML_GP_REGRESSION_DICTIONARY_H
#define ML_GP_REGRESSION_DICTIONARY_H

#include <deque>
#include <vector>
#include "fastlib/fastlib.h"

namespace ml {
class Dictionary {

  private:

    const Matrix *table_;

    std::vector<int> random_permutation_;

    std::deque<bool> in_dictionary_;

    std::vector<int> point_indices_in_dictionary_;

    std::vector<int> training_index_to_dictionary_position_;

    Matrix *current_kernel_matrix_;

    Matrix *current_kernel_matrix_inverse_;

  private:

    void RandomPermutation_(std::vector<int> &permutation);

    void UpdateDictionary_(
      int new_point_index,
      const Vector &temp_kernel_vector,
      double self_kernel_value,
      double projection_error,
      const Vector &inverse_times_kernel_vector);

  public:

    bool in_dictionary(int training_point_index) const {
      return in_dictionary_[training_point_index];
    }

    ~Dictionary() {
      delete current_kernel_matrix_;
      delete current_kernel_matrix_inverse_;
      delete current_kernel_matrix_inverse_row_sum_;
    }

    int position_to_training_index_map(int position) const {
      return random_permutation_[ position ];
    }

    int training_index_to_dictionary_position(int training_index) const {
      return training_index_to_dictionary_position_[training_index];
    }

    int point_indices_in_dictionary(int nth_dictionary_point_index) const {
      return point_indices_in_dictionary_[nth_dictionary_point_index];
    }

    void Init(const Matrix *table_in);

    template<typename KernelType>
    void AddBasis(int iteration_number,
                  const KernelType &kernel);

    const std::vector<int> *basis_set() {
      return &point_indices_in_dictionary_;
    }

    const Matrix &table() const {
      return *table_;
    }

    int size() const;

    Matrix *current_kernel_matrix();

    Matrix *current_kernel_matrix_inverse();
};
};

#endif

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
      const Vector &new_column_vector,
      double self_value,
      double projection_error,
      const Vector &inverse_times_column_vector);

  public:

    Dictionary(const Dictionary &dictionary_in) {
      table_ = dictionary_in.table();
      random_permutation_ = dictionary_in.random_permutation();
      in_dictionary_ = dictionary_in.in_dictionary();
      point_indices_in_dictionary_ =
        dictionary_in.point_indices_in_dictionary();
      training_index_to_dictionary_position_ =
        dictionary_in.training_index_to_dictionary_position();
      current_kernel_matrix_ = new Matrix();
      current_kernel_matrix_->Copy(* dictionary_in.current_kernel_matrix());
      current_kernel_matrix_inverse_ = new Matrix();
      current_kernel_matrix_inverse_->Copy(
        * dictionary_in.current_kernel_matrix_inverse());
    }

    bool in_dictionary(int training_point_index) const {
      return in_dictionary_[training_point_index];
    }

    ~Dictionary() {
      if (current_kernel_matrix_ != NULL) {
        delete current_kernel_matrix_;
      }
      if (current_kernel_matrix_inverse_ != NULL) {
        delete current_kernel_matrix_inverse_;
      }
    }

    Dictionary() {
      table_ = NULL;
      current_kernel_matrix_ = NULL;
      current_kernel_matrix_inverse_ = NULL;
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

    void AddBasis(
      int iteration_number,
      const Vector &new_column_vector,
      double self_value);

    const Matrix *table() const {
      return table_;
    }

    const std::vector<int> &random_permutation() const {
      return random_permutation_;
    }

    const std::deque<bool> &in_dictionary() const {
      return in_dictionary_;
    }

    const std::vector<int> &point_indices_in_dictionary() const {
      return point_indices_in_dictionary_;
    }

    const std::vector<int> &training_index_to_dictionary_position() const {
      return training_index_to_dictionary_position_;
    }

    const Matrix *current_kernel_matrix() const {
      return current_kernel_matrix_;
    }

    const Matrix *current_kernel_matrix_inverse() const {
      return current_kernel_matrix_inverse_;
    }

    int size() const {
      return point_indices_in_dictionary_.size();
    }
};
};

#endif

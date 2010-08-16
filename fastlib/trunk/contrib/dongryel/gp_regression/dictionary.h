/** @file dictionary.h
 *
 *  @brief A generic dictionary for subset of regressor methods.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_GP_REGRESSION_DICTIONARY_H
#define MLPACK_GP_REGRESSION_DICTIONARY_H

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

    void inactive_indices(std::vector<int> *inactive_indices_out) const;

    Dictionary(const Dictionary &dictionary_in);

    bool in_dictionary(int training_point_index) const;

    ~Dictionary();

    Dictionary();

    int position_to_training_index_map(int position) const;

    int training_index_to_dictionary_position(int training_index) const;

    int point_indices_in_dictionary(int nth_dictionary_point_index) const;

    void Init(const Matrix *table_in);

    void AddBasis(
      int new_point_index,
      const Vector &new_column_vector,
      double self_value);

    const Matrix *table() const;

    const std::vector<int> &random_permutation() const;

    const std::deque<bool> &in_dictionary() const;

    const std::vector<int> &point_indices_in_dictionary() const;

    const std::vector<int> &training_index_to_dictionary_position() const;

    const Matrix *current_kernel_matrix() const;

    const Matrix *current_kernel_matrix_inverse() const;

    int size() const;
};
};

#endif

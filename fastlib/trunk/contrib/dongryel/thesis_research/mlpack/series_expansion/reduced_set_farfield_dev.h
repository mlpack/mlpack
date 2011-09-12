/** @file reduced_set_farfield_dev.h
 *
 *  The farfield expansion using the reduced set method.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_SERIES_EXPANSION_REDUCED_SET_FARFIELD_DEV_H
#define MLPACK_SERIES_EXPANSION_REDUCED_SET_FARFIELD_DEV_H

#include <armadillo>
#include "mlpack/series_expansion/dense_matrix_inverse.h"
#include "mlpack/series_expansion/reduced_set_farfield.h"

namespace mlpack {
namespace series_expansion {

template<typename TreeIteratorType>
ReducedSetFarField<TreeIteratorType>::ReducedSetFarField() {
  current_kernel_matrix_ = NULL;
  current_kernel_matrix_inverse_ = NULL;
  num_compressed_points_ = 0;
}

template<typename TreeIteratorType>
ReducedSetFarField<TreeIteratorType>::~ReducedSetFarField() {
  if(current_kernel_matrix_ != NULL) {
    delete current_kernel_matrix_;
  }
  if(current_kernel_matrix_inverse_ != NULL) {
    delete current_kernel_matrix_inverse_;
  }
}

template<typename TreeIteratorType>
void ReducedSetFarField<TreeIteratorType>::UpdateDictionary_(
  const arma::vec &new_point,
  int new_point_index,
  const TreeIteratorType &it,
  const arma::vec &new_column_vector,
  double self_value,
  double projection_error,
  const arma::vec &inverse_times_column_vector) {

  // Add the point to the dictionary.
  std::pair< arma::vec *, int > new_pair(
    new arma::vec(), new_point_index);
  (*new_pair.first) = new_point;
  dictionary_.push_back(new_pair);
  in_dictionary_[ new_point_index - it.begin()] = true;

  // Update the kernel matrix.
  arma::mat *new_kernel_matrix = new arma::mat();
  new_kernel_matrix->set_size(
    current_kernel_matrix_->n_rows + 1, current_kernel_matrix_->n_cols + 1);

  for(unsigned int j = 0; j < current_kernel_matrix_->n_cols; j++) {
    for(unsigned int i = 0; i < current_kernel_matrix_->n_rows; i++) {
      new_kernel_matrix->at(i, j) = current_kernel_matrix_->at(i, j);
    }
  }
  for(unsigned int j = 0; j < current_kernel_matrix_->n_cols; j++) {
    new_kernel_matrix->at(j, current_kernel_matrix_->n_cols) =
      new_column_vector[j];
    new_kernel_matrix->at(current_kernel_matrix_->n_rows, j) =
      new_column_vector[j];
  }

  // Store the self value.
  new_kernel_matrix->at(
    current_kernel_matrix_->n_rows,
    current_kernel_matrix_->n_cols) = self_value;
  delete current_kernel_matrix_;
  current_kernel_matrix_ = new_kernel_matrix;

  // Update the kernel matrix inverse.
  arma::mat *new_kernel_matrix_inverse =
    mlpack::series_expansion::DenseMatrixInverse::Update(
      *current_kernel_matrix_inverse_,
      inverse_times_column_vector,
      projection_error);
  delete current_kernel_matrix_inverse_;
  current_kernel_matrix_inverse_ = new_kernel_matrix_inverse;
}

template<typename TreeIteratorType>
const arma::mat *ReducedSetFarField <
TreeIteratorType >::current_kernel_matrix() const {
  return current_kernel_matrix_;
}

template<typename TreeIteratorType>
const arma::mat *ReducedSetFarField <
TreeIteratorType >::current_kernel_matrix_inverse() const {
  return current_kernel_matrix_inverse_;
}

template<typename TreeIteratorType>
arma::mat *ReducedSetFarField <
TreeIteratorType >::current_kernel_matrix() {
  return current_kernel_matrix_;
}

template<typename TreeIteratorType>
arma::mat *ReducedSetFarField <
TreeIteratorType >::current_kernel_matrix_inverse() {
  return current_kernel_matrix_inverse_;
}

template<typename TreeIteratorType>
void ReducedSetFarField<TreeIteratorType>::AddBasis_(
  const arma::vec &new_point,
  int new_point_index,
  const TreeIteratorType &it,
  const arma::vec &new_column_vector_in,
  double self_value) {

  static const double adding_threshold_ = 1e-5;

  if(new_column_vector_in.n_elem > 0) {

    // Compute the matrix-vector product.
    arma::vec inverse_times_column_vector =
      (*current_kernel_matrix_inverse_) * new_column_vector_in;

    // Compute the projection error.
    double projection_error =
      self_value - arma::dot(new_column_vector_in, inverse_times_column_vector);

    // If the projection error is above the threshold, add it to the
    // dictionary.
    if(projection_error > adding_threshold_) {
      UpdateDictionary_(
        new_point, new_point_index, it, new_column_vector_in, self_value,
        projection_error, inverse_times_column_vector);
    }
  }
  else {

    // Add the point to the dictionary.
    std::pair< arma::vec *, int > new_pair(
      new arma::vec(), new_point_index);
    (*new_pair.first) = new_point;
    dictionary_.push_back(new_pair);
    in_dictionary_[new_point_index - it.begin()] = true;

    // By default, we start with 1 by 1 kernel matrix and inverse.
    current_kernel_matrix_ = new arma::mat();
    current_kernel_matrix_->set_size(1, 1);
    current_kernel_matrix_->at(0, 0) = self_value;
    current_kernel_matrix_inverse_ = new arma::mat();
    current_kernel_matrix_inverse_->set_size(1, 1);
    current_kernel_matrix_inverse_->at(0, 0) = self_value;
  }
}

template<typename TreeIteratorType>
void ReducedSetFarField<TreeIteratorType>::Init(const TreeIteratorType &it) {

}

template<typename TreeIteratorType>
template <
typename MetricType, typename KernelAuxType >
void ReducedSetFarField<TreeIteratorType>::FillKernelValues_(
  const MetricType &metric_in,
  const KernelAuxType &kernel_aux_in,
  const arma::vec &candidate,
  arma::vec *kernel_values_out,
  double *self_value) const {

  // Resize the kernel value vector.
  kernel_values_out->set_size(dictionary_.size());

  // First the self kernel value.
  *self_value = kernel_aux_in.kernel().EvalUnnormOnSq(0.0);

  for(unsigned int i = 0; i < dictionary_.size(); i++) {
    const arma::vec &dictionary_point = *(dictionary_[i].first);
    double squared_distance =
      metric_in.DistanceSq(candidate, dictionary_point);
    (*kernel_values_out)[i] =
      kernel_aux_in.kernel().EvalUnnormOnSq(squared_distance);
  }
}

template<typename TreeIteratorType>
template<typename MetricType, typename KernelAuxType>
void ReducedSetFarField<TreeIteratorType>::FinalizeCompression_(
  const MetricType &metric_in,
  const KernelAuxType &kernel_aux_in,
  TreeIteratorType &it) {

  // Sum of the kernel matrix inverse entries.
  double sum_kernel_matrix_inverse_entries =
    arma::accu(*current_kernel_matrix_inverse_);

  // Column sum of the kernel matrix inverse entries.
  arma::vec column_sum_kernel_matrix_inverse =
    arma::sum(* current_kernel_matrix_inverse_, 1);

  // Compute the projection matrix based on the dictionary.
  it.Reset();
  projection_matrix_.zeros(it.count(), dictionary_.size());
  int num_dictionary_point_encountered = 0;
  while(it.HasNext()) {
    arma::vec point;
    it.Next(&point);

    // The DFS index shifted so that the begin index is 0.
    int adjusted_current_index = it.current_index() - it.begin();

    // If the point is in the dictionary, then set the corresponding
    // column to 1.
    if(in_dictionary_[adjusted_current_index]) {
      projection_matrix_.at(
        adjusted_current_index, num_dictionary_point_encountered) = 1.0;
      num_dictionary_point_encountered++;
    }
    else {
      arma::vec kernel_values;
      double self_value;
      FillKernelValues_(
        metric_in, kernel_aux_in, point, &kernel_values, &self_value);
      arma::vec temp = (* current_kernel_matrix_inverse_) * kernel_values;
      double scale_factor =
        (1.0 - arma::accu(temp)) / sum_kernel_matrix_inverse_entries;
      for(int i = 0; i < projection_matrix_.n_cols(); i++) {
        projection_matrix_.at(
          adjusted_current_index, i) =
            temp[i] + scale_factor * column_sum_kernel_matrix_inverse[i];
      }
    }
  } // end of looping over each point.
}

template<typename TreeIteratorType>
template<typename MetricType, typename KernelAuxType>
void ReducedSetFarField<TreeIteratorType>::AccumulateCoeffs(
  const MetricType &metric_in,
  const KernelAuxType &kernel_aux_in,
  TreeIteratorType &it) {

  // The bit flag for denoting whether each point is in the dictionary
  // or not.
  in_dictionary_.resize(it.count());
  std::fill(in_dictionary_.begin(), in_dictionary_.end(), false);

  // Loop through each point and build the dictionary.
  it.Reset();
  arma::vec new_column_vector_in;
  while(it.HasNext()) {
    arma::vec point;
    it.Next(&point);

    // Fill out the kernel values, and do the self-computation.
    double self_value;
    FillKernelValues_(
      metric_in, kernel_aux_in, point, &new_column_vector_in, &self_value);
    AddBasis_(
      point, it.current_index(), it, new_column_vector_in, self_value);
  } // end of looping over each point.

  // Finalize the compression.
  FinalizeCompression_(metric_in, kernel_aux_in, it);

  // Increment the number of points compressed.
  num_compressed_points_ += it.count();
}

template<typename TreeIteratorType>
template<typename MetricType, typename KernelAuxType>
double ReducedSetFarField<TreeIteratorType>::EvaluateField(
  const MetricType &metric_in,
  const KernelAuxType &kernel_aux_in,
  const arma::vec &query_point) const {

  // Compute the kernel value between the query point and the
  // dictionary point.
  arma::vec kernel_values;
  double self_value;
  FillKernelValues_(
    metric_in, kernel_aux_in, query_point, &kernel_values, &self_value);

  // Go through the projection matrix, and sum up the contribution.
  double contribution = 0.0;
  int num_dictionary_point_encountered = 0;
  for(int i = 0; i < projection_matrix_.n_rows(); i++) {

    // If the point is in the dictionary, then set the corresponding
    // column to 1.
    if(in_dictionary_[i]) {
      contribution += kernel_values[num_dictionary_point_encountered];
      num_dictionary_point_encountered++;
    }
    else {
      for(int j = 0; j < projection_matrix_.n_cols(); j++) {
        contribution += projection_matrix_.at(i, j) * kernel_values[j];
      }
    }
  }
  return contribution;
}

template<typename TreeIteratorType>
int ReducedSetFarField<TreeIteratorType>::num_compressed_points() const {
  return num_compressed_points_;
}

template<typename TreeIteratorType>
template<typename KernelAuxType>
void ReducedSetFarField<TreeIteratorType>::Print(
  const KernelAuxType &kernel_aux_in, const char *name, FILE *stream) const {

}

template<typename TreeIteratorType>
const typename mlpack::series_expansion::
ReducedSetFarField<TreeIteratorType>::DictionaryType &ReducedSetFarField <
TreeIteratorType >::dictionary() const {

  return dictionary_;
}

template<typename TreeIteratorType>
template<typename MetricType, typename KernelAuxType>
void ReducedSetFarField<TreeIteratorType>::TranslateFromFarField(
  const MetricType &metric_in,
  const KernelAuxType &kernel_aux_in,
  const ReducedSetFarField &se,
  TreeIteratorType &it) {

  // Add the pointer to the child expansion.
  child_expansions_.push_back(&se);
  num_compressed_points_ += se.num_compressed_points();

  // Keep adding the pointers to the existing series expansion object,
  // and if we are done adding all, then start compressing all of the
  // dictionaries.
  if(num_compressed_points_ == it.count()) {

    // Take all dictionary points and compress.
    for(int i = 0; i < child_expansions_.size(); i++) {

      // Get the list of dictionary points for the current child
      // expansion.
      const DictionaryType &child_dictionary =
        child_expansions_[i]->dictionary();
      for(unsigned int i = 0; i < child_dictionary.size(); i++) {
        const arma::vec &point = *(child_dictionary[i].first);
        int point_dfs_index = child_dictionary[i].second;

        // Fill out the kernel values, and do the self-computation.
        double self_value;
        arma::vec new_column_vector_in;
        FillKernelValues_(
          metric_in, kernel_aux_in, point, &new_column_vector_in, &self_value);
        AddBasis_(
          point, point_dfs_index, it, new_column_vector_in, self_value);
      } // end of looping over each point.
    }

    // Finalize the compression.
    FinalizeCompression_(metric_in, kernel_aux_in, it);
  }
}

template<typename TreeIteratorType>
template<typename KernelAuxType, typename ReducedSetLocalType>
void ReducedSetFarField<TreeIteratorType>::TranslateToLocal(
  const KernelAuxType &kernel_aux_in, int truncation_order,
  ReducedSetLocalType *se) const {

}
}
}

#endif

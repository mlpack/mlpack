/** @file sample_dense_matrix.h
 *
 *  An abstraction for representing the loading/saving of a subset of
 *  a dense matrix.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_SAMPLE_DENSE_MATRIX_H
#define CORE_TABLE_SAMPLE_DENSE_MATRIX_H

#include "core/table/dense_matrix.h"
#include <boost/serialization/serialization.hpp>

namespace core {
namespace table {

/** @brief A helper class for serializing/unserializing a selected
 *         subset of dense matrix and its associated old_from_new
 *         mapping (see core/tree/general_spacetree.h for details).
 */
template<typename OldFromNewIndexType>
class SampleDenseMatrix {
  private:

    // For boost serialization.
    friend class boost::serialization::access;

  private:

    /** @brief The pointer to the dense matrix to be
     *         serialized/unserialized.
     */
    core::table::DenseMatrix *matrix_;

    /** @brief The associated old from new mapping.
     */
    OldFromNewIndexType *old_from_new_;

    /** @brief The indices to be serialized.
     */
    const std::vector<int> *indices_to_be_serialized_;

    /** @brief The starting column id of the matrix.
     */
    int starting_column_index_;

    /** @brief The number of points to load.
     */
    int num_entries_to_load_;

  public:

    /** @brief
     */
    core::table::DenseMatrix *matrix() {
      return matrix_;
    }

    OldFromNewIndexType *old_from_new() {
      return old_from_new_;
    }

    int starting_column_index() const {
      return starting_column_index_;
    }

    /** @brief The default constructor that initializes every member
     *         to its default value.
     */
    SampleDenseMatrix() {
      matrix_ = NULL;
      old_from_new_ = NULL;
      indices_to_be_serialized_ = NULL;
      starting_column_index_ = 0;
      num_entries_to_load_ = 0;
    }

    /** @brief Call this function before seriaizing (before save
     *         function is called).
     */
    void Init(
      core::table::DenseMatrix &matrix_in,
      OldFromNewIndexType *old_from_new_in,
      const std::vector<int> &indices_to_be_serialized_in) {

      matrix_ = &matrix_in;
      old_from_new_ = old_from_new_in;
      indices_to_be_serialized_ = &indices_to_be_serialized_in;
    }

    /** @brief Call this function before unserializing (before load
     *         function is called).
     */
    void Init(
      core::table::DenseMatrix &matrix_in,
      OldFromNewIndexType *old_from_new_in,
      int starting_column_index_in,
      int num_entries_to_load_in) {
      matrix_ = &matrix_in;
      old_from_new_ = old_from_new_in;
      indices_to_be_serialized_ = NULL;
      starting_column_index_ = starting_column_index_in;
      num_entries_to_load_ = num_entries_to_load_in;
    }

    /** @brief Extract a given list of indices of points along with
     *         its old_from_new mappings onto new destinations.
     */
    void Export(
      core::table::DenseMatrix *matrix_out,
      OldFromNewIndexType *old_from_new_out,
      int starting_column_index_in) const {

      int destination_column_index = starting_column_index_in;
      for(unsigned int i = 0; i < indices_to_be_serialized_->size();
          i++, destination_column_index++) {
        int source_point_index = (*indices_to_be_serialized_)[i];
        core::table::DensePoint source_point;
        matrix_->MakeColumnVector(source_point_index, &source_point);
        core::table::DensePoint destination_point;
        matrix_out->MakeColumnVector(
          destination_column_index, &destination_point);
        for(int j = 0; j < source_point.length(); j++) {
          destination_point[j] = source_point[j];
        }
        old_from_new_out[destination_column_index] =
          old_from_new_[source_point_index];
      }
    }

    /** @brief Serialize a given list of indices of points along with
     *         its old_from_new mappings.
     */
    template<class Archive>
    void save(Archive &ar, const unsigned int version) const {
      for(unsigned int i = 0; i < indices_to_be_serialized_->size(); i++) {
        int point_index = (*indices_to_be_serialized_)[i];
        core::table::DensePoint point;
        matrix_->MakeColumnVector(point_index, &point);
        for(int j = 0; j < point.length(); j++) {
          ar & point[j];
        }
        ar & old_from_new_[point_index].first;
        ar & old_from_new_[point_index].second.first;
      }
    }

    /** @brief Load a pre-specified number of points along with its
     *         old_from_new_mappings, offsetted by a prespecified
     *         position.
     */
    template<class Archive>
    void load(Archive &ar, const unsigned int version) {
      double *ptr = matrix_->GetColumnPtr(starting_column_index_);
      OldFromNewIndexType *old_from_new_ptr =
        old_from_new_ + starting_column_index_;
      for(int i = 0; i < num_entries_to_load_; i++) {
        for(int j = 0; j < matrix_->n_rows(); j++) {
          ar & ptr[j];
        }
        ptr += matrix_->n_rows();

        ar & old_from_new_ptr->first;
        ar & old_from_new_ptr->second.first;
        old_from_new_ptr++;
      }
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};
}
}

#endif

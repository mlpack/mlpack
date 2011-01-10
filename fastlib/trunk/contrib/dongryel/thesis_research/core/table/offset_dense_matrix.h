/** @file offset_dense_matrix.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_OFFSET_DENSE_MATRIX_H
#define CORE_TABLE_OFFSET_DENSE_MATRIX_H

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

class OffsetDenseMatrix {
  private:
    friend class boost::serialization::access;

    /** @brief The raw pointer to the set of points.
     */
    double *ptr_;

    /** @brief The pointer to the old from new mappings.
     */
    std::pair<int, std::pair<int, int> > *index_ptr_;

    /** @brief The process ID to which each point is assigned.
     */
    std::vector<int> *assignment_indices_;

    /** @brief The serialization process selects the point IDs with
     *         the assignment index equal to this value.
     */
    int filter_index_;

    /** @brief The process ID of the calling process.
     */
    int rank_;

    /** @brief The dimensionality of each point.
     */
    int n_attributes_;

    /** @brief The total number of points. This is equal to the length
     *         of assignment_indices_.
     */
    int n_entries_;

  private:

    int Count_() const {
      int num_doubles = 0;
      for(unsigned int i = 0; i < assignment_indices_->size(); i++) {
        if((*assignment_indices_)[i] == filter_index_) {
          num_doubles++;
        }
      }

      // Add two for sending the process ID and the point ID.
      num_doubles *= (n_attributes_ + 2);
      return num_doubles;
    }

  public:

    int n_attributes() const {
      return n_attributes_;
    }

    int n_entries() const {
      return n_entries_;
    }

    OffsetDenseMatrix() {
      ptr_ = NULL;
      index_ptr_ = NULL;
      assignment_indices_ = NULL;
      filter_index_ = -1;
      rank_ = -1;
      n_attributes_ = -1;
      n_entries_ = -1;
    }

    void Init(
      int rank_in, double *ptr_in,
      std::pair<int, std::pair< int, int > > *index_ptr_in,
      int n_attributes_in) {
      ptr_ = ptr_in;
      index_ptr_ = index_ptr_in;
      n_attributes_ = n_attributes_in;
      rank_ = rank_in;
    }

    void Init(
      int rank_in,
      core::table::DenseMatrix &mat_in,
      std::pair<int, std::pair< int, int > > *index_ptr_in,
      std::vector<int> &assignment_indices_in,
      int filter_index_in) {
      ptr_ = mat_in.ptr();
      index_ptr_ = index_ptr_in;
      n_attributes_ = mat_in.n_rows();
      assignment_indices_ = &assignment_indices_in;
      filter_index_ = filter_index_in;
      rank_ = rank_in;
    }

    void Extract(
      double *ptr_out, std::pair<int, std::pair<int, int> > *index_ptr_out) {
      int num_doubles = Count_();
      n_entries_ = num_doubles / (n_attributes_ + 2);

      // Loop through and find out the columns to serialize.
      double *ptr_iter = ptr_;
      for(unsigned int i = 0; i < assignment_indices_->size();
          i++, ptr_iter += n_attributes_) {
        if((*assignment_indices_)[i] == filter_index_) {
          for(int j = 0; j < n_attributes_; j++) {
            ptr_out[j] = ptr_iter[j];
          }
          *index_ptr_out =
            std::pair <
            int, std::pair<int, int> > (
              filter_index_, std::pair<int, int>(i, 0));
          index_ptr_out++;
          ptr_out += n_attributes_;
        }
      }
    }

    template<class Archive>
    void save(Archive &ar, const unsigned int version) const {

      // First, save the number of doubles to be serialized.
      int num_doubles = Count_();
      ar & num_doubles;

      // Loop through and find out the columns to serialize.
      double *ptr_iter = ptr_;
      for(unsigned int i = 0; i < assignment_indices_->size();
          i++, ptr_iter += n_attributes_) {
        if((*assignment_indices_)[i] == filter_index_) {

          // Write out the point.
          for(int j = 0; j < n_attributes_; j++) {
            ar & ptr_iter[j];
          }

          // Write out the process ID and the point ID.
          ar & rank_;
          ar & i;
        }
      }
    }

    template<class Archive>
    void load(Archive &ar, const unsigned int version) {

      // Load the number of points to be unfrozen.
      int num_doubles;
      ar & num_doubles;
      n_entries_ = num_doubles / (n_attributes_ + 2);
      double *ptr_iter = ptr_;
      for(int i = 0; i < n_entries_; i++, ptr_iter += n_attributes_) {
        for(int j = 0; j < n_attributes_; j++) {
          ar & ptr_iter[j];
        }
        ar & index_ptr_[i].first;
        ar & index_ptr_[i].second.first;
        index_ptr_[i].second.second = 0;
      }
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};
};
};

#endif

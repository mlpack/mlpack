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
class OffsetDenseMatrix {
  private:
    friend class boost::serialization::access;

    double *ptr_;

    std::pair<int, int> *index_ptr_;

    std::vector<int> *assignment_indices_;

    int filter_index_;

    int n_attributes_;

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
      n_attributes_ = -1;
      n_entries_ = -1;
    }

    void Init(
      double *ptr_in, std::pair<int, int> *index_ptr_in, int n_attributes_in) {
      ptr_ = ptr_in;
      index_ptr_ = index_ptr_in;
      n_attributes_ = n_attributes_in;
    }

    void Init(
      core::table::DenseMatrix &mat_in,
      std::pair<int, int> *index_ptr_in,
      std::vector<int> &assignment_indices_in,
      int filter_index_in) {
      ptr_ = mat_in.ptr();
      index_ptr_ = index_ptr_in;
      n_attributes_ = mat_in.n_rows();
      assignment_indices_ = &assignment_indices_in;
      filter_index_ = filter_index_in;
    }

    void Extract(double *ptr_out, std::pair<int, int> *index_ptr_out) {
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
          *index_ptr_out = std::pair<int, int>(filter_index_, i);
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
          ar & filter_index_;
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
        ar & index_ptr_[i].second;
      }
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};
};
};

#endif

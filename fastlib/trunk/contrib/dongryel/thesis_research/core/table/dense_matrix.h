/** @file dense_matrix.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_DENSE_MATRIX_H
#define CORE_TABLE_DENSE_MATRIX_H

#include "dense_point.h"
#include "memory_mapped_file.h"
#include <boost/interprocess/offset_ptr.hpp>
#include <boost/serialization/serialization.hpp>

namespace core {
namespace table {
class DenseMatrix {

  private:
    friend class boost::serialization::access;

    boost::interprocess::offset_ptr<double> ptr_;

    int n_rows_;

    int n_cols_;

  public:

    template<class Archive>
    void save(Archive &ar, const unsigned int version) const {
      ar & n_rows_;
      ar & n_cols_;
      int num_elements = n_rows_ * n_cols_;
      for(int i = 0; i < num_elements; i++) {
        ar & ptr_.get()[i];
      }
    }

    template<class Archive>
    void load(Archive &ar, const unsigned int version) {
      ar & n_rows_;
      ar & n_cols_;
      int num_elements = n_rows_ * n_cols_;
      ptr_ = (core::table::global_m_file_) ?
             core::table::global_m_file_->ConstructArray<double>(num_elements) :
             new double[num_elements];
      for(int i = 0; i < num_elements; i++) {
        ar & ptr_[i];
      }
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    void Print() const {
      for(int i = 0; i < n_rows_; i++) {
        for(int j = 0; j < n_cols_; j++) {
          printf("%g, ", this->get(i, j));
        }
        printf("\n");
      }
    }

    void SetZero() {
      memset(ptr_.get(), 0, sizeof(double) * n_rows_ * n_cols_);
    }

    const double *ptr() const {
      return ptr_.get();
    }

    double *ptr() {
      return ptr_.get();
    }

    int n_rows() const {
      return n_rows_;
    }

    int n_cols() const {
      return n_cols_;
    }

    void swap_cols(int first_col, int second_col) {
      double *first_ptr = ptr_.get() + first_col * n_rows_;
      double *second_ptr = ptr_.get() + second_col * n_rows_;
      for(int i = 0; i < n_rows_; first_ptr++, second_ptr++, i++) {
        std::swap(*first_ptr, *second_ptr);
      }
    }

    void set(int row, int col, double val) {
      ptr_[col * n_rows_ + row] = val;
    }

    double get(int row, int col) const {
      return ptr_[col * n_rows_ + row];
    }

    void Reset() {
      ptr_ = NULL;
      n_rows_ = n_cols_ = 0;
    }

    DenseMatrix() {
      Reset();
    }

    ~DenseMatrix() {
      if(ptr_.get() != NULL) {
        if(core::table::global_m_file_) {
          core::table::global_m_file_->DestroyPtr(ptr_.get());
        }
        else {
          delete[] ptr_.get();
        }
      }
      Reset();
    }

    void Init(
      int n_rows_in, int n_cols_in) {

      ptr_ = (core::table::global_m_file_) ?
             core::table::global_m_file_->ConstructArray<double>(
               n_rows_in * n_cols_in) :
             new double[n_rows_in * n_cols_in];
      n_rows_ = n_rows_in;
      n_cols_ = n_cols_in;
    }

    const double *GetColumnPtr(int column_id) const {
      return ptr_.get() + column_id * n_rows_;
    }

    double *GetColumnPtr(int column_id) {
      return ptr_.get() + column_id * n_rows_;
    }

    void CopyColumnVector(int column_id, double *point_out) const {
      const double *ptr = ptr_.get() + column_id * n_rows_;
      for(int i = 0; i < n_rows_; ptr++, i++) {
        point_out[i] = (*ptr);
      }
    }

    void MakeColumnVector(int column_id, std::vector<double> *point_out) const {
      point_out->resize(n_rows_);
      const double *ptr = ptr_.get() + column_id * n_rows_;
      for(int i = 0; i < n_rows_; ptr++, i++) {
        (*point_out)[i] = (*ptr);
      }
    }

    void MakeColumnVector(
      int i, core::table::DensePoint *point_out) const {
      point_out->Alias(
        ptr_.get() + i * n_rows_, n_rows_);
    }

    void MakeColumnVector(int i, core::table::DensePoint *point_out) {
      point_out->Alias(
        ptr_.get() + i * n_rows_, n_rows_);
    }
};
};
};

#endif

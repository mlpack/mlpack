/** @file dense_matrix.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_DENSE_MATRIX_H
#define CORE_TABLE_DENSE_MATRIX_H

#include <armadillo>
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

    bool is_alias_;

  public:

    bool is_alias() const {
      return is_alias_;
    }

    void operator=(const DenseMatrix &dense_matrix_in) {

      // Not a true copy constructor. Sets the matrix to be an alias.
      ptr_ = const_cast<double *>(dense_matrix_in.ptr());
      n_rows_ = dense_matrix_in.n_rows();
      n_cols_ = dense_matrix_in.n_cols();
      is_alias_ = true;
    }

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
      if(is_alias_ == false) {
        ptr_ =
          (core::table::global_m_file_) ?
          core::table::global_m_file_->ConstructArray<double>(num_elements) :
          new double[num_elements];
      }
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
      is_alias_ = false;
    }

    DenseMatrix() {
      Reset();
    }

    ~DenseMatrix() {
      if(is_alias_ == false) {
        if(ptr_.get() != NULL) {
          if(core::table::global_m_file_) {
            core::table::global_m_file_->DestroyPtr(ptr_.get());
          }
          else {
            delete[] ptr_.get();
          }
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

    void MakeColumnVector(int column_id, double *point_out) const {
      const double *ptr = ptr_.get() + column_id * n_rows_;
      for(int i = 0; i < n_rows_; ptr++, i++) {
        point_out[i] = (*ptr);
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

    void MakeColumnVector(
      int i, arma::vec *vec_out) const {
      const_cast<arma::u32 &>(vec_out->n_rows) = n_rows_;
      const_cast<arma::u32 &>(vec_out->n_cols) = 1;
      const_cast<arma::u32 &>(vec_out->n_elem) = n_rows_;
      const_cast<bool &>(vec_out->use_aux_mem) = true;
      const_cast<double *&>(vec_out->mem) =
        const_cast<double *>(ptr_.get() + i * n_rows_);
    }

    void MakeColumnVector(int i, arma::vec *vec_out) {
      const_cast<arma::u32 &>(vec_out->n_rows) = n_rows_;
      const_cast<arma::u32 &>(vec_out->n_cols) = 1;
      const_cast<arma::u32 &>(vec_out->n_elem) = n_rows_;
      const_cast<bool &>(vec_out->use_aux_mem) = true;
      const_cast<double *&>(vec_out->mem) =
        const_cast<double *>(ptr_.get() + i * n_rows_);
    }

    void Alias(const double *ptr_in, int n_rows_in, int n_cols_in) {
      ptr_ = const_cast<double *>(ptr_in);
      n_rows_ = n_rows_in;
      n_cols_ = n_cols_in;
      is_alias_ = true;
    }
};
};
};

#endif

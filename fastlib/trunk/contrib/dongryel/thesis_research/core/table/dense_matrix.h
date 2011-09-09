/** @file dense_matrix.h
 *
 *  A namespace containing utilities for a dense column-oriented
 *  matrix.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_DENSE_MATRIX_H
#define CORE_TABLE_DENSE_MATRIX_H

#include <armadillo>
#include <boost/interprocess/offset_ptr.hpp>
#include <boost/serialization/serialization.hpp>
#include "dense_point.h"
#include "memory_mapped_file.h"

namespace core {
namespace table {

extern core::table::MemoryMappedFile *global_m_file_;

/** @brief The function that takes a raw pointer and creates an alias
 *         armadillo vector.
 */
template<typename T>
static void PtrToArmaMat(
  const T *matrix_in, int n_rows_in, int n_cols_in, arma::Mat<T> *mat_out) {

  // This constructor uses the const_cast for a hack. For some reason,
  // Armadillo library does not allow creation of aliases for const
  // T type pointers, so I used const_cast here.
  const_cast<arma::u32 &>(mat_out->n_rows) = n_rows_in;
  const_cast<arma::u32 &>(mat_out->n_cols) = n_cols_in;
  const_cast<arma::u32 &>(mat_out->n_elem) = n_rows_in * n_cols_in;
  const_cast<arma::u16 &>(mat_out->vec_state) = 0;
  const_cast<arma::u16 &>(mat_out->mem_state) = 2;
  const_cast<T *&>(mat_out->mem) = const_cast<T *>(matrix_in);
}

template<typename T>
void MakeColumnVector(
  const arma::Mat<T> &mat_in, int column_index, arma::Col<T> *col_out) {

  core::table::PtrToArmaVec(
    mat_in.memptr() + column_index * mat_in.n_rows, mat_in.n_rows, col_out);
}

template<typename T>
T *GetColumnPtr(const arma::Mat<T> &mat_in, int column_index) {

  return const_cast< arma::Mat<T> &>(mat_in).memptr() + column_index *
         mat_in.n_rows;
}
}
}

namespace boost {
namespace serialization {

template<class Archive, class T>
inline void serialize(
  Archive & ar,
  arma::Mat<T> & t,
  const unsigned int file_version) {
  split_free(ar, t, file_version);
}

template<class Archive, class T>
void save(Archive & ar, const arma::Mat<T> & t, unsigned int version) {

  // First save the dimensions.
  ar & t.n_rows;
  ar & t.n_cols;
  for(unsigned int j = 0; j < t.n_cols; j++) {
    for(unsigned int i = 0; i < t.n_rows; i++) {
      T val = t.at(i, j);
      ar & val;
    }
  }
}

template<class Archive, class T>
void load(Archive & ar, arma::Mat<T> & t, unsigned int version) {

  // Load the dimensions.
  int n_rows;
  int n_cols;
  ar & n_rows;
  ar & n_cols;

  // The new memory block.
  T *new_ptr =
    (core::table::global_m_file_) ?
    core::table::global_m_file_->ConstructArray<T>(n_rows * n_cols) :
    new T[n_rows * n_cols];
  int pos = 0;
  for(int j = 0; j < n_cols; j++) {
    for(int i = 0; i < n_rows; i++, pos++) {
      ar & new_ptr[pos];
    }
  }

  // Finally, put the memory block into the armadillo matrix.
  core::table::PtrToArmaMat(new_ptr, n_rows, n_cols, &t);
}
}
}

#endif

/** @file dense_matrix.h
 *
 *  A wrapper on the Armadillo matrix.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_DENSE_MATRIX_H
#define CORE_TABLE_DENSE_MATRIX_H

#include "core/table/dense_point.h"

namespace core {
namespace table {
class DenseMatrix: public virtual arma::mat {
  public:

    void MakeColumnVector(int column_id, std::vector<double> *point_out) const {
      point_out->resize(this->n_rows);
      const double *ptr = this->memptr() + column_id * this->n_rows;
      for(int i = 0; i < this->n_rows; ptr++, i++) {
        (*point_out)[i] = (*ptr);
      }
    }

    void MakeColumnVector(
      int i, core::table::DenseConstPoint *point_out) const {
      point_out->Alias(
        this->memptr() + i * this->n_rows, this->n_rows);
    }

    void MakeColumnVector(int i, core::table::DensePoint *point_out) {
      point_out->Alias(
        this->memptr() + i * this->n_rows, this->n_rows);
    }
};
};
};

#endif

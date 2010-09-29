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

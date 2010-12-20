/** @file linear_algebra.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_MATH_LINEAR_ALGEBRA_H
#define CORE_MATH_LINEAR_ALGEBRA_H

#include <armadillo>

namespace core {
namespace table {
class DensePoint;
class DenseMatrix;
};
};

namespace core {
namespace math {

template<typename MatrixType>
static void MatrixTripleProduct(
  const MatrixType &left, const MatrixType &mid,
  core::table::DenseMatrix *product) {

  product->Init(left.n_rows(), left.n_cols());

  // Use armadillo matrices to compute the triple product. This makes
  // a light copy, so there is very little performance lost.
  arma::mat *left_copy = new arma::mat(
    left.ptr(), left.n_rows(), left.n_cols());
  arma::mat *mid_copy = new arma::mat(mid.ptr(), mid.n_rows(), mid.n_cols());
  arma::mat product_alias(
    product->ptr(), product->n_rows(), product->n_cols(), false);
  product_alias = (*left_copy) * (*mid_copy) * arma::trans(*left_copy);

  delete left_copy;
  delete mid_copy;
}

template<typename MatrixType>
static void MatrixTripleProduct(
  const MatrixType &left, const MatrixType &mid, const MatrixType &right,
  core::table::DenseMatrix *product) {

  product->Init(left.n_rows(), right.n_cols());

  // Use armadillo matrices to compute the triple product. This makes
  // a light copy, so there is very little performance lost.
  arma::mat *left_copy = new arma::mat(
    left.ptr(), left.n_rows(), left.n_cols());
  arma::mat *mid_copy = new arma::mat(mid.ptr(), mid.n_rows(), mid.n_cols());
  arma::mat *right_copy = new arma::mat(
    right.ptr(), right.n_rows(), right.n_cols());
  arma::mat product_alias(
    product->ptr(), product->n_rows(), product->n_cols(), false);
  product_alias = (*left_copy) * (*mid_copy) * (*right_copy);

  delete left_copy;
  delete mid_copy;
  delete right_copy;
}

template<typename VectorType>
static double Dot(const VectorType &a, const VectorType &b) {
  arma::mat a_mat(a.ptr(), a.length());
  arma::mat b_mat(b.ptr(), b.length());
  return arma::dot(a_mat, b_mat);
}

template<typename VectorType>
static double LengthEuclidean(const VectorType &a) {
  return sqrt(core::math::Dot(a, a));
}

template<typename VectorType>
static void CopyValues(
  const VectorType &vec_in, VectorType *vec_out) {

  for(unsigned int i = 0; i < vec_in.n_elem; i++) {
    (*vec_out)[i] = vec_in[i];
  }
}

template<typename VectorType>
static void SubFrom(
  const VectorType &vec_in, VectorType *vec_out) {

  for(unsigned int i = 0; i < vec_in.n_elem; i++) {
    (*vec_out)[i] -= vec_in[i];
  }
}

template<typename VectorType>
static void SubInit(
  const VectorType &sub, const VectorType &sub_from, VectorType *vec_out) {
  vec_out->Init(sub.length());
  for(int i = 0; i < sub.length(); i++) {
    (*vec_out)[i] = sub_from[i] - sub[i];
  }
}

template<typename VectorType>
static void AddExpert(
  double scale, const VectorType &vec_scaled, VectorType *vec_add_to) {

  for(int i = 0; i < vec_scaled.length(); i++) {
    (*vec_add_to)[i] += scale * vec_scaled[i];
  }
}

template<typename VectorType>
static void AddTo(
  const VectorType &vec_in, VectorType *vec_out) {

  for(unsigned int i = 0; i < vec_in.n_elem; i++) {
    (*vec_out)[i] += vec_in[i];
  }
}

template<typename VectorType>
class ScaleTrait {
  public:
    static void Compute(double scale, VectorType *vec);
};

template<>
class ScaleTrait<core::table::DensePoint> {
  public:
    static void Compute(double scale, core::table::DensePoint *vec) {
      for(int i = 0; i < vec->length(); i++) {
        (*vec)[i] *= scale;
      }
    }
};

template<>
class ScaleTrait<core::table::DenseMatrix> {
  public:
    static void Compute(double scale, core::table::DenseMatrix *vec) {
      for(int j = 0; j < vec->n_cols(); j++) {
        for(int i = 0; i < vec->n_rows(); i++) {
          vec->set(i, j, vec->get(i, j) * scale);
        }
      }
    }
};

template<typename VectorType>
static void Scale(double scale, VectorType *vec) {
  ScaleTrait<VectorType>::Compute(scale, vec);
}

template<typename VectorType>
static void ScaleOverwrite(
  double scale, const VectorType &vec_in, VectorType *vec_out) {

  for(unsigned int i = 0; i < vec_in.n_elem; i++) {
    (*vec_out)[i] = vec_in[i] * scale;
  }
}

template<typename VectorType>
static void ScaleInit(
  double scale, const VectorType &vec_in, VectorType *vec_out) {

  vec_out->set_size(vec_in.n_elem);
  ScaleOverwrite(scale, vec_in, vec_out);
}
};
};

#endif

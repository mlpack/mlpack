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
  arma::mat left_alias(left.ptr(), left.n_rows(), left.n_cols());
  arma::mat mid_alias(mid.ptr(), mid.n_rows(), mid.n_cols());
  arma::mat product_alias(
    product->ptr(), product->n_rows(), product->n_cols(), false);
  product_alias = left_alias * mid_alias * arma::trans(left_alias);
}

template<typename MatrixType>
static void MatrixTripleProduct(
  const MatrixType &left, const MatrixType &mid, const MatrixType &right,
  core::table::DenseMatrix *product) {

  product->Init(left.n_rows(), right.n_cols());

  // Use armadillo matrices to compute the triple product. This makes
  // a light copy, so there is very little performance lost.
  arma::mat left_alias(left.ptr(), left.n_rows(), left.n_cols());
  arma::mat mid_alias(mid.ptr(), mid.n_rows(), mid.n_cols());
  arma::mat right_alias(right.ptr(), right.n_rows(), right.n_cols());
  arma::mat product_alias(
    product->ptr(), product->n_rows(), product->n_cols(), false);
  product_alias = left_alias * mid_alias * right_alias;
}

template<typename VectorType>
static double Dot(const VectorType &a, const VectorType &b) {
  arma::vec a_mat(a.ptr(), a.length());
  arma::vec b_mat(b.ptr(), b.length());
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
static void SubOverwrite(
  const VectorType &sub, const VectorType &sub_from, VectorType *vec_out) {
  for(int i = 0; i < sub.length(); i++) {
    (*vec_out)[i] = sub_from[i] - sub[i];
  }
}

template<typename VectorType>
static void SubInit(
  const VectorType &sub, const VectorType &sub_from, VectorType *vec_out) {
  vec_out->Init(sub.length());
  SubOverwrite(sub, sub_from, vec_out);
}

template<typename T>
class AddExpertTrait {
  public:
    static void Compute(double scale, const T &vec_scaled, T *vec_add_to);
};

template<>
class AddExpertTrait<core::table::DenseMatrix> {
  public:
    static void Compute(
      double scale, const core::table::DenseMatrix &mat_scaled,
      core::table::DenseMatrix *mat_add_to) {

      arma::mat mat_scaled_alias(
        mat_scaled.ptr(), mat_scaled.n_rows(), mat_scaled.n_cols());
      arma::mat mat_add_to_alias(
        mat_add_to->ptr(), mat_add_to->n_rows(), mat_add_to->n_cols(), false);
      mat_add_to_alias = mat_add_to_alias + scale * mat_scaled_alias;
    }
};

template<>
class AddExpertTrait<core::table::DensePoint> {
  public:
    static void Compute(
      double scale, const core::table::DensePoint &vec_scaled,
      core::table::DensePoint *vec_add_to) {
      arma::vec vec_scaled_alias(
        vec_scaled.ptr(), vec_scaled.length());
      arma::vec vec_add_to_alias(
        vec_add_to->ptr(), vec_add_to->length(), false);
      vec_add_to_alias = vec_add_to_alias + scale * vec_scaled_alias;
    }
};

template<typename T>
static void AddExpert(
  double scale, const T &vec_scaled, T *vec_add_to) {
  core::math::AddExpertTrait<T>::Compute(scale, vec_scaled, vec_add_to);
}

template<typename VectorType>
static void AddTo(
  const VectorType &vec_in, VectorType *vec_out) {

  arma::vec vec_in_alias(vec_in.ptr(), vec_in.length());
  arma::vec vec_out_alias(vec_out->ptr(), vec_in.length(), false);
  vec_out_alias = vec_out_alias + vec_in_alias;
}

/** @brief Computes $c = c + \alpha * a b^T$.
 */
template<typename VectorType, typename MatrixType>
static void MulExpert(
  double alpha, const VectorType &a, const VectorType &b, MatrixType *c) {

  arma::vec a_alias(a.ptr(), a.length());
  arma::vec b_alias(b.ptr(), b.length());
  arma::mat c_alias(c->ptr(), c->n_rows(), c->n_cols(), false);
  c_alias = c_alias + alpha * a_alias * arma::trans(b_alias);
}

template<typename MatrixType, typename VectorType>
static void MulInit(
  const MatrixType &a, const VectorType &b, VectorType *c) {

  arma::mat a_alias(a.ptr(), a.n_rows(), a.n_cols());
  arma::vec b_alias(b.ptr(), b.length());
  c->Init(a.n_rows());
  arma::vec c_alias(c->ptr(), c.length(), false);
  c_alias = a_alias * b_alias;
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
  arma::vec vec_in_alias(vec_in.ptr(), vec_in.length());
  arma::vec vec_out_alias(vec_out->ptr(), vec_in.length(), false);
  for(unsigned int i = 0; i < vec_in_alias.n_elem; i++) {
    vec_out_alias[i] = vec_in_alias[i] * scale;
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

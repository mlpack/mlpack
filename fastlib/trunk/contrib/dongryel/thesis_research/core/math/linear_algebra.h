/** @file linear_algebra.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_MATH_LINEAR_ALGEBRA_H
#define CORE_MATH_LINEAR_ALGEBRA_H

#include <armadillo>

namespace core {
namespace math {

template<typename VectorType>
static double Dot(const VectorType &a, const VectorType &b) {
  double dot_product = 0;
  for(int i = 0; i < a.length(); i++) {
    dot_product += a[i] * b[i];
  }
  return dot_product;
}

static void CopyValues(const arma::vec &vec_in, arma::vec *vec_out);

static void SubFrom(
  const arma::vec &vec_in, arma::vec *vec_out);

static void AddTo(
  const arma::vec &vec_in, arma::vec *vec_out);

static void ScaleOverwrite(
  double scale, const arma::vec &vec_in, arma::vec *vec_out);

static void ScaleInit(
  double scale, const arma::vec &vec_in, arma::vec *vec_out);
};
};

#endif

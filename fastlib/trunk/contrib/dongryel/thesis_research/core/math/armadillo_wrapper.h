/** @file armadillo_wrapper.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_MATH_ARMADILLO_WRAPPER_H
#define CORE_MATH_ARMADILLO_WRAPPER_H

#include <armadillo>

namespace core {
namespace math {
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

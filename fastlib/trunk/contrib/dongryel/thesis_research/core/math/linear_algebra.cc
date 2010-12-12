/** @file linear_algebra.cc
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include "linear_algebra.h"

namespace core {
namespace math {
static void CopyValues(
  const arma::vec &vec_in, arma::vec *vec_out) {

  for(int i = 0; i < vec_in.n_elem; i++) {
    (*vec_out)[i] = vec_in[i];
  }
}

static void SubFrom(
  const arma::vec &vec_in, arma::vec *vec_out) {

  for(int i = 0; i < vec_in.n_elem; i++) {
    (*vec_out)[i] -= vec_in[i];
  }
}

static void AddTo(
  const arma::vec &vec_in, arma::vec *vec_out) {

  for(int i = 0; i < vec_in.n_elem; i++) {
    (*vec_out)[i] += vec_in[i];
  }
}

static void ScaleOverwrite(
  double scale, const arma::vec &vec_in, arma::vec *vec_out) {

  for(int i = 0; i < vec_in.n_elem; i++) {
    (*vec_out)[i] = vec_in[i] * scale;
  }
}

static void ScaleInit(
  double scale, const arma::vec &vec_in, arma::vec *vec_out) {

  vec_out->set_size(vec_in.n_elem);
  ScaleOverwrite(scale, vec_in, vec_out);
}
};
};

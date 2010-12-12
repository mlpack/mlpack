/** @file armadillo_wrapper.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include "armadillo_wrapper.h"

namespace core {
namespace math {
static void ScaleInit(
  double scale, const arma::vec &vec_in, arma::vec *vec_out) {

  vec_out->set_size(vec_in.n_elem);
  for(int i = 0; i < vec_in.n_elem; i++) {
    (*vec_out)[i] = vec_in[i] * scale;
  }
}
};
};

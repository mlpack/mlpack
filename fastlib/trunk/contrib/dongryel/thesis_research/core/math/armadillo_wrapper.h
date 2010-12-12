/** @file armadillo_wrapper.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_MATH_ARMADILLO_WRAPPER_H
#define CORE_MATH_ARMADILLO_WRAPPER_H

#include <armadillo>

namespace core {
namespace math {
static void ScaleInit(
  double scale, const arma::vec &vec_in, arma::vec *vec_out);
};
};

#endif

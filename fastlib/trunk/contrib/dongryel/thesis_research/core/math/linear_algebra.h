/** @file linear_algebra.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_MATH_LINEAR_ALGEBRA_H
#define CORE_MATH_LINEAR_ALGEBRA_H

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
static void Scale(double scale, VectorType *vec) {
  for(unsigned int i = 0; i < vec->length(); i++) {
    (*vec)[i] *= scale;
  }
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

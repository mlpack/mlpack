#ifndef UPPER_TRIANGULAR_SQUARE_MATRIX_H
#define UPPER_TRIANGULAR_SQUARE_MATRIX_H

#include "fastlib/fastlib.h"

/** @brief The class which provides a wrapper class such that the
 *         vector is used as an upper triangular square matrix.
 */
template<int num_rows>
class UpperTriangularSquareMatrix {
  
 private:
  index_t position_(index_t r, index_t c) const {
    return (num_rows - 1) * num_rows / 2 -
      (num_rows - r - 1) * (num_rows - r) / 2 + (c - r - 1); 
  }
  
 public:
  double vector_[(num_rows - 1) * num_rows / 2];
 
  double get(index_t r, index_t c) const {
    index_t r_trans = std::min(r, c);
    index_t c_trans = std::max(r, c);   
    DEBUG_ASSERT(r_trans < c_trans);
    return vector_[position_(r_trans, c_trans)];
  }
  
  void set(index_t r, index_t c, double value) {
    index_t r_trans = std::min(r, c);
    index_t c_trans = std::max(r, c);
    vector_[position_(r_trans, c_trans)] = value;
  }
 
  static inline const int total_size() {
    return (num_rows - 1) * num_rows / 2;
  }

  void Init() {
    vector_.Init(total_size());
    SetZero();
  }
  
  template<int t_numerator, int t_denominator>
  void Pow(UpperTriangularSquareMatrix &result) const {
    for(index_t i = 0; i < UpperTriangularSquareMatrix::total_size(); i++) {
      result.vector_[i] = math::Pow<t_numerator, t_denominator>
	(vector_[i]);
    }
  }

  void SetZero() {
    vector_.SetZero();
  }  
};

#endif

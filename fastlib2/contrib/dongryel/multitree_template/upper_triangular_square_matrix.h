#ifndef UPPER_TRIANGULAR_SQUARE_MATRIX_H
#define UPPER_TRIANGULAR_SQUARE_MATRIX_H

/** @brief The class which provides a wrapper class such that the
 *         vector is used as an upper triangular square matrix.
 */
class UpperTriangularSquareMatrix {
  
 private:
  index_t position_(index_t r, index_t c) const {
    return (num_rows_ - 1) * (num_rows_) / 2 -
      (num_rows_ - r - 1) * (num_rows_ - r) / 2 + (c - r - 1); 
  }
  
 public:
  Vector vector_;
  
  int num_rows_;
  
  void Init(int num_rows) {
    
    num_rows_ = num_rows;
    index_t total_size = (num_rows - 1) * num_rows / 2;
    vector_.Init(total_size);
    SetZero();
  }
  
  void SetZero() {
    vector_.SetZero();
  }

  double get(index_t r, index_t c) const {
    DEBUG_ASSERT(r < c);
    return vector_[position_(r, c)];
  }
  
  void set(index_t r, index_t c, double value) {
    vector_[position_(r, c)] = value;
  }
};

#endif

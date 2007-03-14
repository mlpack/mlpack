// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file matrix.h
 *
 * Basic double-precision vector and matrix classes.
 */

#ifndef LA_MATRIX_H
#define LA_MATRIX_H

#include "base/common.h"
#include "base/scale.h"
#include "base/cc.h"
#include "base/ccmem.h"

#include <stdlib.h>
#include <string.h>
#include <cmath>

/**
 * Double-precision vector for use with LAPACK.
 *
 * This supports aliasing, so you can have weak copies of a vector,
 * or weak copies to subsections of a vector (or weak copies to a column
 * of a matrix).
 *
 * Vectors were never meant to support resizing, nor was it meant to hold
 * anything but floating-point values.  For a suitable structure, see
 * ArrayList.
 *
 * @code
 * Vector orig;
 * orig.Init(5);
 * for (index_t i = 0; i < 5; i++) {
 *   orig[i] = 2.0;
 * }
 * Vector an_alias;
 * an_alias.Alias(orig);
 * an_alias[4] = 99;
 * assert(orig[4] == 9);
 * @endcode
 */
class Vector {
 private:
  /** The pointer to the array of doubles. */
  double *ptr_;
  /** The length of the vector. */
  index_t length_;
  /** Whether this should be freed, i.e. it is not an alias. */
  bool should_free_;
  
 public:
  /**
   * Creates a completely uninitialized Vector which is only useful for
   * transferring ownership to.
   */
  Vector() {
    DEBUG_ONLY(Uninitialize_());
  }
  
  /**
   * Copy constructor -- for use in collections.
   */
  Vector(const Vector& other) {
    DEBUG_ONLY(Uninitialize_());
    Copy(other);
  }
  CC_ASSIGNMENT_OPERATOR(Vector);
  
  /**
   * Destroys the Vector, freeing the memory if this copy is not an alias.
   */
  ~Vector() {
    Destruct();
  }
  
  /**
   * Uninitializes so that you can call another initializer.
   */
  void Destruct() {
    DEBUG_ASSERT_MSG(ptr_ != BIG_BAD_POINTER(double),
       "You forgot to initialize a Vector before it got automatically freed.");
    
    /* mark slow case as "unlikely" even if it might be the likely case */
    if (unlikely(should_free_)) {
      mem::DebugPoison(ptr_, length_);
      mem::Free(ptr_);
    }
    
    DEBUG_ONLY(Uninitialize_());
  }

  /**
   * Creates a vector of a particular length, but does not initialize the
   * values in it.
   */
  void Init(index_t in_length) {
    ptr_ = mem::Alloc<double>(in_length);
    length_ = in_length;
    should_free_ = true;
  }
  
  /**
   * Sets all elements to the same value.
   */
  void SetAll(double d) {
    mem::ConstructAll(ptr_, d, length_);
  }
  
  /**
   * Sets all elements to zero.
   */
  void SetZero() {
    // TODO: if IEEE is used, this can be done efficiently with memset
    SetAll(0);
  }

  /**
   * Makes this uninitialized vector a copy of the other vector.
   *
   * @param other the vector to explicitly copy
   */
  void Copy(const Vector& other) {
    Copy(other.ptr(), other.length());
  }

  /**
   * Makes this uninitialized vector a copy of the other vector.
   *
   * @param doubles the array of doubles to copy
   * @param in_length the number of doubles in the array
   */
  void Copy(const double *doubles, index_t in_length) {
    DEBUG_ONLY(AssertUninitialized_());
    
    ptr_ = mem::Dup(doubles, in_length);
    length_ = in_length;
    should_free_ = true;
  }
  
  /**
   * Alias a particular memory region of doubles.
   */
  void Alias(double *in_ptr, index_t in_length) {
    DEBUG_ONLY(AssertUninitialized_());
    
    ptr_ = in_ptr;
    length_ = in_length;
    should_free_ = false;
  }
  
  /**
   * Implements the "Copiable" interface using .
   */
  void WeakCopy(const Vector& other) {
    Alias(other);
  }
  
  /**
   * Makes this vector an alias of another vector.
   *
   * @param other the other vector
   */
  void Alias(const Vector& other) {
    // we trust in good faith that a const vector won't be abused
    Alias(other.ptr_, other.length());
  }
  
  /**
   * Makes this vector the "owning copy" of the other vector; the other
   * vector becomes an alias and this becomes the standard.
   *
   * @param other a pointer to the vector whose contents will be owned
   */
  void Own(Vector* other) {
    Own(other->ptr_, other->length());
    
    DEBUG_ASSERT(other->should_free_);
    other->should_free_ = false;
  }
  
  /**
   * Become owner of a particular pointer in memory that was allocated
   * with mem::Alloc<double>.
   */
  void Own(double *in_ptr, index_t in_length) {
    DEBUG_ONLY(AssertUninitialized_());
    
    ptr_ = in_ptr;
    length_ = in_length;
    should_free_ = true;
  }
  
  template<typename Serializer>
  void Serialize(Serializer *s) const {
    s->Put(length_);
    s->Put(ptr_, length_);
  }
  
  template<typename Deserializer>
  void Deserialize(Deserializer *s) {
    DEBUG_ONLY(AssertUninitialized_());
    s->Get(&length_);
    ptr_ = mem::Alloc<double>(length_);
    s->Get(ptr_, length_);
    should_free_ = true;
  }
  
  /**
   * Initializes an uninitialized vector as an alias to a a sub-region
   * of this vector.
   *
   * @param start_index the first index
   * @param len the length
   * @param dest an UNINITIALIZED vector to use
   */
  void MakeSubvector(index_t start_index, index_t len, Vector* dest) {
    DEBUG_BOUNDS(start_index, length_);
    DEBUG_BOUNDS(start_index + len - 1, length_);
    
    dest->Alias(ptr_ + start_index, len);
  }
  
  /**
   * Swaps all values in this vector with values in the other.
   *
   * This is different from Swap, because Swap will only change what these
   * point to.
   *
   * @param other an identically sized vector to swap values with
   */
  void SwapValues(Vector* other) {
    DEBUG_ASSERT(length() == other->length());
    mem::Swap(ptr_, other->ptr_, length_);
  }
  
  /**
   * Copies the values from another matrix to this matrix.
   *
   * @param other the vector to copy from
   */
  void CopyValues(const Vector& other) {
    DEBUG_ASSERT(length() == other.length());
    mem::Copy(ptr_, other.ptr_, length_);
  }
  
 public:
  index_t length() const {
    return length_;
  }
  
  double *ptr() {
    return ptr_;
  }
  
  const double *ptr() const {
    return ptr_;
  }
  
  double operator [] (index_t i) const {
    DEBUG_BOUNDS(i, length_);
    return ptr_[i];
  }
  
  double &operator [] (index_t i) {
    DEBUG_BOUNDS(i, length_);
    return ptr_[i];
  }
  
 private:
  void AssertUninitialized_() const {
    DEBUG_ASSERT(length_ == BIG_BAD_NUMBER);
  }
  
  void Uninitialize_() {
    DEBUG_ONLY(ptr_ = BIG_BAD_POINTER(double));
    DEBUG_ONLY(length_ = BIG_BAD_NUMBER);
  }
  
  void AssertInitialized_() {
    DEBUG_ASSERT_MSG(ptr_ != BIG_BAD_POINTER(double),
        "Vector was not initialized.");
  }
};

/**
 * Double-precision column-major matrix for use with LAPACK.
 *
 * Your code can have huge performance hits if you fail to realize this
 * is column major.  For datasets, your columns should be individual points
 * and your rows should be features.
 *
 * TODO: If it's not entirely obvious or well documented how to use this
 * class please let the FASTlib people know.
 */
class Matrix {
 private:
  /** Linearized matrix (column-major). */
  double *ptr_;
  /** Number of rows. */
  index_t n_rows_;
  /** Number of columns. */
  index_t n_cols_;
  /** Whether I am a strong copy (not an alias). */
  bool should_free_;

 public:
  /**
   * Creates a Matrix with uninitialized elements of the specified size.
   */
  Matrix(index_t in_rows, index_t in_cols) {
    DEBUG_ONLY(Uninitialize_());
    Init(in_rows, in_cols);
  }

  /**
   * Copy constructor -- for use in collections.
   */
  Matrix(const Matrix& other) {
    DEBUG_ONLY(Uninitialize_());
    Copy(other);
  }
  CC_ASSIGNMENT_OPERATOR(Matrix);

  /**
   * Non-initializing constructor.
   */
  Matrix() {
    DEBUG_ONLY(Uninitialize_());
  }

  /**
   * Empty destructor.
   */
  ~Matrix() {
    Destruct();
  }
  
  /**
   * Destructs this, so that it is suitable for you to call an initializer
   * on this again.
   */
  void Destruct() {
    DEBUG_ASSERT_MSG(ptr_ != BIG_BAD_POINTER(double),
       "You forgot to initialize a Matrix before it got automatically freed.");
    if (unlikely(should_free_)) {
      mem::DebugPoison(ptr_, n_rows_ * n_cols_);
      mem::Free(ptr_);
      DEBUG_ONLY(Uninitialize_());
    }
    DEBUG_POISON_PTR(ptr_);
    DEBUG_ONLY(n_rows_ = BIG_BAD_NUMBER);
    DEBUG_ONLY(n_cols_ = BIG_BAD_NUMBER);
  }

  /**
   * Creates a Matrix with uninitialized elements of the specified size.
   */
  void Init(index_t in_rows, index_t in_cols) {
    DEBUG_ONLY(AssertUninitialized_());
    ptr_ = mem::Alloc<double>(in_rows * in_cols);
    n_rows_ = in_rows;
    n_cols_ = in_cols;
    should_free_ = true;
  }

  /**
   * Sets the entire matrix to zero.
   */
  void SetAll(double d) {
    mem::ConstructAll(ptr_, d, n_elements());
  }

  /**
   * Makes this matrix all zeroes.
   */
  void SetZero() {
    // TODO: If IEEE floating point is used, this can just be a memset to
    // zero
    SetAll(0);
  }

  /**
   * Makes this uninitialized matrix a copy of the other vector.
   *
   * @param other the vector to explicitly copy
   */
  void Copy(const Matrix& other) {
    Copy(other.ptr(), other.n_rows(), other.n_cols());    
  }

  /**
   * Makes this uninitialized matrix a copy of the other vector.
   *
   * @param ptr_in the pointer to a block of column-major doubles
   * @param n_rows_in the number of rows
   * @param n_cols_in the number of columns
   */
  void Copy(const double *ptr_in, index_t n_rows_in, index_t n_cols_in) {
    DEBUG_ONLY(AssertUninitialized_());
    
    ptr_ = mem::Dup(ptr_in, n_rows_in * n_cols_in);
    n_rows_ = n_rows_in;
    n_cols_ = n_cols_in;
    should_free_ = true;
  }
  
  /**
   * Makes this uninitialized matrix an alias of another matrix.
   *
   * Changes to one matrix are visible in the other (and vice-versa).
   *
   * @param other the other vector
   */
  void Alias(const Matrix& other) {
    // we trust in good faith that const-ness won't be abused
    Alias(other.ptr_, other.n_rows(), other.n_cols());
  }
  
  /**
   * Makes this uninitialized matrix an alias of an existing block of doubles.
   *
   * @param ptr_in the pointer to a block of column-major doubles
   * @param n_rows_in the number of rows
   * @param n_cols_in the number of columns
   */
  void Alias(double *ptr_in, index_t n_rows_in, index_t n_cols_in) {
    DEBUG_ONLY(AssertUninitialized_());
    
    ptr_ = ptr_in;
    n_rows_ = n_rows_in;
    n_cols_ = n_cols_in;
    should_free_ = false;
  }
  
  /**
   * Makes this a weak copy or alias of the other.
   *
   * This is identical to Alias.
   */
  void WeakCopy(const Matrix& other) {
    Alias(other);
  }
  
  /**
   * Makes this uninitialized matrix the "owning copy" of the other matrix;
   * the other vector becomes an alias and this becomes the standard.
   *
   * The other matrix must be the "owning" copy of its memory.
   *
   * @param other a pointer to the other matrix
   */
  void Own(Matrix* other) {
    Own(other->ptr(), other->n_rows(), other->n_cols());
    
    DEBUG_ASSERT(other->should_free_);
    other->should_free_ = false;
  }
  
  /**
   * Initializes this uninitialized matrix as the "owning copy" of some
   * linearized chunk of RAM allocated with mem::Alloc.
   *
   * @param ptr_in the pointer to a block of column-major doubles
   *        allocated via Mem::Alloc
   * @param n_rows_in the number of rows
   * @param n_cols_in the number of columns
   */
  void Own(double *ptr_in, index_t n_rows_in, index_t n_cols_in) {
    DEBUG_ONLY(AssertUninitialized_());
    
    ptr_ = ptr_in;
    n_rows_ = n_rows_in;
    n_cols_ = n_cols_in;
    should_free_ = true;
  }
  
  template<typename Serializer>
  void Serialize(Serializer *s) const {
    s->Put(n_rows_);
    s->Put(n_cols_);
    s->Put(ptr_, n_elements());
  }
  
  template<typename Deserializer>
  void Deserialize(Deserializer *s) {
    DEBUG_ONLY(AssertUninitialized_());
    s->Get(&n_rows_);
    s->Get(&n_cols_);
    ptr_ = mem::Alloc<double>(n_elements());
    s->Get(ptr_, n_elements());
    should_free_ = true;
  }
  
  /**
   * Make a matrix that is an alias of a particular slice of my columns.
   *
   * @param dest an UNINITIALIZED matrix
   */
  void MakeColumnSlice(index_t start_col, index_t n_cols_new,
      Matrix *dest) const {
    DEBUG_BOUNDS(start_col, n_cols_);
    DEBUG_BOUNDS(start_col + n_cols_new - 1, n_cols_);
    dest->Alias(ptr_ + start_col * n_rows_,
        n_rows_, n_cols_new);
  }
  
  /**
   * Make an alias of a reshaped version of this matrix (column-major format).
   *
   * For instance, a matrix with 2 rows and 6 columns can be reshaped
   * into a matrix with 12 rows 1 column, 1 row and 12 columns, or a variety
   * of other shapes.  The layout of the new elements correspond exactly
   * to just pretending that the current column-major matrix is laid out
   * as a different column-major matrix.
   *
   * It is required that n_rows_new * n_cols_new is the same as
   * n_rows * n_cols of the original matrix.
   *
   * TODO: Considering using const Matrix& for third-party classes that want
   * to implicitly convert to Matrix.
   *
   * @param n_rows_in new number of rows
   * @param n_cols_in new number of columns
   * @param dest a pointer to an unitialized matrix
   * @return a reshaped matrix backed by the original
   */
  void MakeReshaped(index_t n_rows_in, index_t n_cols_in,
      Matrix *dest) const {
    DEBUG_ASSERT(n_rows_in * n_cols_in == n_rows() * n_cols());
    dest->Alias(ptr_, n_rows_in, n_cols_in);
  }
  
  /**
   * Makes an alias of a particular column.
   *
   * @param col the column to alias
   * @param dest a pointer to an uninitialized vector, which will be
   *        initialized as an alias to the particular column
   */
  void MakeColumnVector(index_t col, Vector *dest) const {
    DEBUG_BOUNDS(col, n_cols_);
    dest->Alias(n_rows_ * col + ptr_, n_rows_);
  }
  
  /**
   * Retrieves a pointer to a contiguous array corresponding to a particular
   * column.
   *
   * @param col the column number
   * @return an array where the i'th element is the i'th row of that
   *         particular column
   */
  double *GetColumnPtr(index_t col) {
    DEBUG_BOUNDS(col, n_cols_);
    return n_rows_ * col + ptr_;
  }
  
  /**
   * Retrieves a pointer to a contiguous array corresponding to a particular
   * column.
   *
   * @param col the column number
   * @return an array where the i'th element is the i'th row of that
   *         particular column
   */
    const double *GetColumnPtr(index_t col) const {
    DEBUG_BOUNDS(col, n_cols_);
    return n_rows_ * col + ptr_;
  }
  
  /**
   * Reduces the number of columns, but REQUIRES that there are no aliases
   * to this matrix anywhere else.
   *
   * @param new_n_cols the new number of columns
   */
  void OwnerReduceColumns(index_t new_n_cols) {
    DEBUG_ASSERT(should_free_); // the best assert we can do
    n_cols_ = new_n_cols;
    ptr_ = mem::Resize(ptr_, n_elements());
  }
  
  /**
   * Swaps all values in this matrix with values in the other.
   *
   * This is different from Swap, because Swap will only change what these
   * point to.
   *
   * @param other an identically sized vector to swap values with
   */
  void SwapValues(Matrix* other) {
    DEBUG_ASSERT(n_cols() == other->n_cols());
    DEBUG_ASSERT(n_rows() == other->n_rows());
    mem::Swap(ptr_, other->ptr_, n_elements());
  }
  
 public:
  /**
   * Returns a pointer to the very beginning of the matrix, stored
   * in a column-major format.
   *
   * This is suitable for BLAS and LAPACK calls.
   */
  const double *ptr() const {
    return ptr_;
  }
  
  /**
   * Returns a pointer to the very beginning of the matrix, stored
   * in a column-major format.
   *
   * This is suitable for BLAS and LAPACK calls.
   */
  double *ptr() {
    return ptr_;
  }
  
  /**
   * Gets a particular double at the specified row and column.
   *
   * @param r the row number
   * @param c the column number
   */
  double get(index_t r, index_t c) const {
    DEBUG_BOUNDS(r, n_rows_);
    DEBUG_BOUNDS(c, n_cols_);
    return ptr_[c * n_rows_ + r];
  }
 
  /**
   * Sets the value at the row and column.
   *
   * @param r the row number
   * @param c the column number
   * @param v the value to set
   */ 
  void set(index_t r, index_t c, double v) {
    DEBUG_BOUNDS(r, n_rows_);
    DEBUG_BOUNDS(c, n_cols_);
    ptr_[c * n_rows_ + r] = v;
  }
  
  /**
   * Gets a reference to a particular row and column.
   *
   * It is highly recommended you treat this as a single value rather than
   * part of an array; use ColumnSlice or Reshaped instead to make
   * subsections.
   */
  double &ref(index_t r, index_t c) {
    DEBUG_BOUNDS(r, n_rows_);
    DEBUG_BOUNDS(c, n_cols_);
    return ptr_[c * n_rows_ + r];
  }
  
  /** Returns the number of columns. */
  index_t n_cols() const {
    return n_cols_;
  }
  
  /** Returns the number of rows. */
  index_t n_rows() const {
    return n_rows_;
  }
  
  /**
   * Returns the total number of elements (power user).
   *
   * This is useful for iterating over all elements of the matrix when the
   * row/column structure is not important.
   */
  size_t n_elements() const {
    // TODO: putting the size_t on the outside may be faster (32-bit
    // versus 64-bit multiplication in cases) but is more likely to result
    // in bugs
    return size_t(n_rows_) * size_t(n_cols_);
  }
  
 private:
  void AssertUninitialized_() const {
    DEBUG_ASSERT(n_rows_ == BIG_BAD_NUMBER);
  }
  
  void Uninitialize_() {
    DEBUG_POISON_PTR(ptr_);
    DEBUG_ONLY(n_rows_ = BIG_BAD_NUMBER);
    DEBUG_ONLY(n_cols_ = BIG_BAD_NUMBER);
  } 

};

#endif

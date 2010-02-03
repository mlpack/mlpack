/* MLPACK 0.2
 *
 * Copyright (c) 2008, 2009 Alexander Gray,
 *                          Garry Boyer,
 *                          Ryan Riegel,
 *                          Nikolaos Vasiloglou,
 *                          Dongryeol Lee,
 *                          Chip Mappus, 
 *                          Nishant Mehta,
 *                          Hua Ouyang,
 *                          Parikshit Ram,
 *                          Long Tran,
 *                          Wee Chin Wong
 *
 * Copyright (c) 2008, 2009 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
/**
 * @file matrix.h
 *
 * Basic double-precision vector and matrix classes.
 */

#ifndef LA_MATRIX_H
#define LA_MATRIX_H

#include "base/base.h"
#ifndef DISABLE_DISK_MATRIX
#include "mmanager/memory_manager.h"
#endif

#include <stdlib.h>
#include <string.h>
#include <math.h>

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
template<typename T>
class GenVector {
 private:
  /** The pointer to the array of doubles. */
  T *ptr_;
  /** The length of the vector. */
  index_t length_;
  /** Whether this should be freed, i.e. it is not an alias. */
  bool should_free_;
  
  OBJECT_TRAVERSAL_ONLY(GenVector) {
    OT_OBJ(length_);
    OT_ALLOC(ptr_, length_);
  }
  OT_REFILL_TRANSIENTS(GenVector) {
    should_free_ = true;
  }
  
 public:
  /**
   * Creates a completely uninitialized Vector which must be initialized.
   */
  GenVector() {
    DEBUG_ONLY(Uninitialize_());
  }
  
  /**
   * Copy constructor -- for use in collections.
   */
  GenVector(const GenVector& other) {
    DEBUG_ONLY(Uninitialize_());
    Copy(other);
  }
  ASSIGN_VIA_COPY_CONSTRUCTION(GenVector);
  
  /**
   * Destroys the Vector, freeing the memory if this copy is not an alias.
   */
  ~GenVector() {
    Destruct();
  }
  
  /**
   * Uninitializes so that you can call another initializer.
   */
  void Destruct() {
    DEBUG_ASSERT_MSG(ptr_ != BIG_BAD_POINTER(T),
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
    ptr_ = mem::Alloc<T>(in_length);
    length_ = in_length;
    should_free_ = true;
  }
  
  /**
   * Creates a vector of a particular length statically, but does not
   * initialize the values in it. This vector will not be freed!
   */
  void StaticInit(index_t in_length) {
#ifndef DISABLE_DISK_MATRIX
    ptr_ = mmapmm::MemoryManager<false>::allocator_->Alloc<T>(in_length);
    length_ = in_length;
    should_free_ = false;
#else
    Init(in_length);
#endif
  }

  /**
   * Sets all elements to the same value.
   */
  void SetAll(T d) {
    mem::RepeatConstruct(ptr_, d, length_);
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
  void Copy(const GenVector& other) {
    Copy(other.ptr(), other.length());
  }

  /**
   * Makes this uninitialized vector a copy of the other vector.
   *
   * @param doubles the array of doubles to copy
   * @param in_length the number of doubles in the array
   */
  void Copy(const T *doubles, index_t in_length) {
    DEBUG_ONLY(AssertUninitialized_());
    
    ptr_ = mem::AllocCopy(doubles, in_length);
    length_ = in_length;
    should_free_ = true;
  }

  /**
   * Makes this uninitialized vector a static copy of the other
   * vector. This copy will not be freed!
   *
   * @param other the vector to explicitly copy
   */
  void StaticCopy(const GenVector& other) {
    StaticCopy(other.ptr(), other.length());
  }

  /**
   * Makes this uninitialized vector a static copy of the other
   * vector. This copy will not be freed!
   *
   * @param doubles the array of doubles to copy
   * @param in_length the number of doubles in the array
   */
  void StaticCopy(const T *doubles, index_t in_length) {
#ifndef DISABLE_DISK_MATRIX
    DEBUG_ONLY(AssertUninitialized_());
    
    ptr_ = mmapmm::MemoryManager<false>::allocator_->Alloc<T>(in_length);
    mem::Copy<T, T, T>(ptr_, doubles, in_length);
    length_ = in_length;
    should_free_ = false;
#else
    Copy(doubles, in_length);
#endif
  }
  
  /**
   * Alias a particular memory region of doubles.
   */
  void Alias(T *in_ptr, index_t in_length) {
    DEBUG_ONLY(AssertUninitialized_());
    
    ptr_ = in_ptr;
    length_ = in_length;
    should_free_ = false;
  }
  
  /**
   * Implements the "Copiable" interface using .
   */
  void WeakCopy(const GenVector& other) {
    Alias(other);
  }
  
  /**
   * Makes this vector an alias of another vector.
   *
   * @param other the other vector
   */
  void Alias(const GenVector& other) {
    // we trust in good faith that a const vector won't be abused
    Alias(other.ptr_, other.length());
  }
  
  /**
   * Makes this vector the "owning copy" of the other vector; the other
   * vector becomes an alias and this becomes the standard.
   *
   * The other vector must be the "owning copy" of its memory.
   *
   * @param other a pointer to the vector whose contents will be owned
   */
  void Own(GenVector* other) {
    Own(other->ptr_, other->length());
    
    DEBUG_ASSERT(other->should_free_);
    other->should_free_ = false;
  }
  
  /**
   * Become owner of a particular pointer in memory that was allocated
   * with mem::Alloc<double>.
   */
  void Own(T *in_ptr, index_t in_length) {
    DEBUG_ONLY(AssertUninitialized_());
    
    ptr_ = in_ptr;
    length_ = in_length;
    should_free_ = true;
  }
  
  /**
   * Makes this vector the "owning copy" of the other vector; the other
   * vector becomes an alias and this becomes the standard statically.
   *
   * The other vector must have been allocated statically.
   *
   * @param other a pointer to the vector whose contents will be owned
   */
  void StaticOwn(GenVector* other) {
#ifndef DISABLE_DISK_MATRIX
    StaticOwn(other->ptr_, other->length());
#else
    Own(other);
#endif
  }
  
  /**
   * Become owner of a particular pointer in memory that was allocated
   * statically.
   */
  void StaticOwn(T *in_ptr, index_t in_length) {
#ifndef DISABLE_DISK_MATRIX
    DEBUG_ONLY(AssertUninitialized_());
    
    ptr_ = in_ptr;
    length_ = in_length;
    should_free_ = false;
#else
    Own(in_ptr, in_length);
#endif
  }
  
  /**
   * Initializes an uninitialized vector as an alias to a a sub-region
   * of this vector.
   *
   * @param start_index the first index
   * @param len the length
   * @param dest an UNINITIALIZED vector to use
   */
  void MakeSubvector(index_t start_index, index_t len, GenVector* dest) {
    DEBUG_BOUNDS(start_index, length_);
    DEBUG_BOUNDS(start_index + len, length_ + 1);
    
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
  void SwapValues(GenVector* other) {
    DEBUG_ASSERT(length() == other->length());
    mem::Swap(ptr_, other->ptr_, length_);
  }
  
  /**
   * Copies the values from another vector to this vector.
   *
   * @param other the vector to copy from
   */
  void CopyValues(const GenVector& other) {
    DEBUG_ASSERT(length() == other.length());
    mem::Copy(ptr_, other.ptr_, length_);
  }

  /**
   * Copies all of the values from an array of doubles to this vector.
   *
   * @param src_ptr the vector to copy from, must have at least
   *        length() elements
   */
  void CopyValues(const T *src_ptr) {
    mem::Copy(ptr_, src_ptr, length_);
  }
  
  /**
   * Prints to a stream as a debug message.
   *
   * @param name a name that will be printed with the vector
   * @param stream the stream to print to, such as stderr (default) or stdout
   */
  void PrintDebug(const char *name = "", FILE *stream = stderr) const {
    fprintf(stream, "----- VECTOR %s ------\n", name);
    for (index_t i = 0; i < length(); i++) {
      fprintf(stream, "%+3.3f ", get(i));
    }
    fprintf(stream, "\n");
  }
  
 public:
  /** The number of elements in this vector. */
  index_t length() const {
    return length_;
  }
  
  /**
   * A pointer to the C-style array containing the elements of this vector.
   */
  T *ptr() {
    return ptr_;
  }
  
  /**
   * A pointer to the C-style array containing the elements of this vector.
   */
  const T *ptr() const {
    return ptr_;
  }
  
  /**
   * Gets the i'th element of this vector.
   */
  T operator [] (index_t i) const {
    DEBUG_BOUNDS(i, length_);
    return ptr_[i];
  }
  
  /**
   * Gets a mutable reference to the i'th element of this vector.
   */
  T &operator [] (index_t i) {
    DEBUG_BOUNDS(i, length_);
    return ptr_[i];
  }
  
  /**
   * Gets a value to the i'th element of this vector (convenient when
   * you have a pointer to a vector).
   *
   * This is identical to the array subscript operator, except for the
   * following reason:
   *
   * @code
   * void FooBar(Vector *v) {
   *    v->get(0) // much easier to read than (*v)[0]
   * }
   * @endcode
   */
  T get(index_t i) const {
    DEBUG_BOUNDS(i, length_);
    return ptr_[i];
  }
  
 private:
  void AssertUninitialized_() const {
    DEBUG_ASSERT_MSG(length_ == BIG_BAD_NUMBER, "Cannot re-init vectors.");
  }
  
  void Uninitialize_() {
    DEBUG_ONLY(ptr_ = BIG_BAD_POINTER(T));
    DEBUG_ONLY(length_ = BIG_BAD_NUMBER);
  }
  
  void AssertInitialized_() {
    DEBUG_ASSERT_MSG(ptr_ != BIG_BAD_POINTER(T),
        "Vector was not initialized.");
  }
};

/** @brief A Vector is a GenVector of double's.
 */
typedef GenVector<double> Vector;

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
template<typename T>
class GenMatrix {
 private:
  /** Linearized matrix (column-major). */
  T *ptr_;
  /** Number of rows. */
  index_t n_rows_;
  /** Number of columns. */
  index_t n_cols_;
  /** Whether I am a strong copy (not an alias). */
  bool should_free_;

  OBJECT_TRAVERSAL_ONLY(GenMatrix) {
    OT_OBJ(n_rows_);
    OT_OBJ(n_cols_);
    OT_ALLOC(ptr_, n_elements());
  }
  OT_REFILL_TRANSIENTS(GenMatrix) {
    should_free_ = false;
  }

 public:
  /**
   * Creates a Matrix with uninitialized elements of the specified size.
   */
  GenMatrix(index_t in_rows, index_t in_cols) {
    DEBUG_ONLY(Uninitialize_());
    Init(in_rows, in_cols);
  }

  /**
   * Copy constructor -- for use in collections.
   */
  GenMatrix(const GenMatrix<T>& other) {
    DEBUG_ONLY(Uninitialize_());
    Copy(other);
  }
  ASSIGN_VIA_COPY_CONSTRUCTION(GenMatrix);

  /**
   * Creates a matrix that can be initialized.
   */
  GenMatrix() {
    DEBUG_ONLY(Uninitialize_());
  }

  /**
   * Empty destructor.
   */
  ~GenMatrix() {
    Destruct();
  }
  
  /**
   * Destructs this, so that it is suitable for you to call an initializer
   * on this again.
   */
  void Destruct() {
    DEBUG_ASSERT_MSG(ptr_ != BIG_BAD_POINTER(T),
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
    ptr_ = mem::Alloc<T>(in_rows * in_cols);
    n_rows_ = in_rows;
    n_cols_ = in_cols;
    should_free_ = true;
  }

  /**
   * Creates a diagonal matrix.
   */
  void InitDiagonal(const GenVector<T>& v) {
    Init(v.length(), v.length());
    SetDiagonal(v);
  }

  /**
   * Creates a Matrix with uninitialized elements of the specified
   * size statically. This matrix is not freed!
   */
  void StaticInit(index_t in_rows, index_t in_cols) {
#ifndef DISABLE_DISK_MATRIX
    DEBUG_ONLY(AssertUninitialized_());
    ptr_ = mmapmm::MemoryManager<false>::allocator_->Alloc<T>
      (in_rows * in_cols);
    n_rows_ = in_rows;
    n_cols_ = in_cols;
    should_free_ = false;
#else
    Init(in_rows, in_cols);
#endif
  }

  /**
   * Creates a diagonal matrix.
   */
  void StaticInitDiagonal(const GenVector<T>& v) {
    StaticInit(v.length(), v.length());
    SetDiagonal(v);
  }

  /**
   * Sets the entire matrix to zero.
   */
  void SetAll(T d) {
    mem::RepeatConstruct(ptr_, d, n_elements());
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
   * Makes this a diagonal matrix whose diagonals are the values in v.
   */
  void SetDiagonal(const GenVector<T>& v) {
    DEBUG_ASSERT(n_rows() == v.length());
    DEBUG_ASSERT(n_cols() == v.length());
    SetZero();
    index_t n = v.length();
    for (index_t i = 0; i < n; i++) {
      set(i, i, v[i]);
    }
  }

  /**
   * Makes this uninitialized matrix a copy of the other vector.
   *
   * @param other the vector to explicitly copy
   */
  void Copy(const GenMatrix& other) {
    Copy(other.ptr(), other.n_rows(), other.n_cols());    
  }

  /**
   * Makes this uninitialized matrix a copy of the other vector.
   *
   * @param ptr_in the pointer to a block of column-major doubles
   * @param n_rows_in the number of rows
   * @param n_cols_in the number of columns
   */
  void Copy(const T *ptr_in, index_t n_rows_in, index_t n_cols_in) {
    DEBUG_ONLY(AssertUninitialized_());
    
    ptr_ = mem::AllocCopy(ptr_in, n_rows_in * n_cols_in);
    n_rows_ = n_rows_in;
    n_cols_ = n_cols_in;
    should_free_ = true;
  }

  /**
   * Makes this uninitialized matrix a static copy of the other
   * vector which will not be freed!
   *
   * @param other the vector to explicitly copy
   */
  void StaticCopy(const GenMatrix& other) {
    StaticCopy(other.ptr(), other.n_rows(), other.n_cols());    
  }

  /**
   * Makes this uninitialized matrix a static copy of the other
   * vector, which will not be freed!
   *
   * @param ptr_in the pointer to a block of column-major doubles
   * @param n_rows_in the number of rows
   * @param n_cols_in the number of columns
   */
  void StaticCopy(const T *ptr_in, index_t n_rows_in, index_t n_cols_in) {
#ifndef DISABLE_DISK_MATRIX
    DEBUG_ONLY(AssertUninitialized_());

    ptr_ = mmapmm::MemoryManager<false>::allocator_->Alloc<T>
      (n_rows_in * n_cols_in);
    mem::Copy<T, T, T>(ptr_, ptr_in, n_rows_in * n_cols_in);

    n_rows_ = n_rows_in;
    n_cols_ = n_cols_in;
    should_free_ = false;
#else
    Copy(ptr_in, n_rows_in, n_cols_in);
#endif
  }
 
  /**
   * Makes this uninitialized matrix an alias of another matrix.
   *
   * Changes to one matrix are visible in the other (and vice-versa).
   *
   * @param other the other vector
   */
  void Alias(const GenMatrix& other) {
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
  void Alias(T *ptr_in, index_t n_rows_in, index_t n_cols_in) {
    DEBUG_ONLY(AssertUninitialized_());
    
    ptr_ = ptr_in;
    n_rows_ = n_rows_in;
    n_cols_ = n_cols_in;
    should_free_ = false;
  }
  
  /**
   * Makes this a 1 row by N column alias of a vector of length N.
   *
   * @param row_vector the vector to alias
   */
  void AliasRowVector(const GenVector<T>& row_vector) {
    Alias(const_cast<T*>(row_vector.ptr()), 1, row_vector.length());
  }
  
  /**
   * Makes this an N row by 1 column alias of a vector of length N.
   *
   * @param col_vector the vector to alias
   */
  void AliasColVector(const GenVector<T>& col_vector) {
    Alias(const_cast<T*>(col_vector.ptr()), col_vector.length(), 1);
  }
  
  /**
   * Makes this a weak copy or alias of the other.
   *
   * This is identical to Alias.
   */
  void WeakCopy(const GenMatrix& other) {
    Alias(other);
  }
  
  /**
   * Makes this uninitialized matrix the "owning copy" of the other
   * matrix; the other vector becomes an alias and this becomes the
   * standard.
   *
   * The other matrix must be the "owning copy" of its memory.
   *
   * @param other a pointer to the other matrix
   */
  void Own(GenMatrix* other) {
    Own(other->ptr(), other->n_rows(), other->n_cols());
    
    DEBUG_ASSERT(other->should_free_);
    other->should_free_ = false;
  }
  
  /**
   * Initializes this uninitialized matrix as the "owning copy" of
   * some linearized chunk of RAM allocated with mem::Alloc.
   *
   * @param ptr_in the pointer to a block of column-major doubles
   *        allocated via mem::Alloc
   * @param n_rows_in the number of rows
   * @param n_cols_in the number of columns
   */
  void Own(T *ptr_in, index_t n_rows_in, index_t n_cols_in) {
    DEBUG_ONLY(AssertUninitialized_());
    
    ptr_ = ptr_in;
    n_rows_ = n_rows_in;
    n_cols_ = n_cols_in;
    should_free_ = true;
  }

  /**
   * Makes this uninitialized matrix the "owning copy" of the other
   * matrix; the other vector becomes an alias and this becomes the
   * standard statically.
   *
   * The other matrix must have been allocated statically.
   *
   * @param other a pointer to the other matrix
   */
  void StaticOwn(GenMatrix* other) {
#ifndef DISABLE_DISK_MATRIX
    StaticOwn(other->ptr(), other->n_rows(), other->n_cols());
#else
    Own(other);
#endif
  }
  
  /**
   * Initializes this uninitialized matrix as the "owning copy" of
   * some linearized chunk of RAM allocated statically.
   *
   * @param ptr_in the pointer to a block of column-major doubles
   *        allocated via mem::Alloc
   * @param n_rows_in the number of rows
   * @param n_cols_in the number of columns
   */
  void StaticOwn(T *ptr_in, index_t n_rows_in, index_t n_cols_in) {
#ifndef DISABLE_DISK_MATRIX
    DEBUG_ONLY(AssertUninitialized_());
    
    ptr_ = ptr_in;
    n_rows_ = n_rows_in;
    n_cols_ = n_cols_in;
    should_free_ = false;
#else
    Own(ptr_in, n_rows_in, n_cols_in);
#endif
  }

  /**
   * Make a matrix that is an alias of a particular slice of my columns.
   *
   * @param start_col the first column
   * @param n_cols_new the number of columns in the new matrix
   * @param dest an UNINITIALIZED matrix
   */
  void MakeColumnSlice(index_t start_col, index_t n_cols_new,
      GenMatrix *dest) const {
    DEBUG_BOUNDS(start_col, n_cols_);
    DEBUG_BOUNDS(start_col + n_cols_new, n_cols_ + 1);
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
      GenMatrix *dest) const {
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
  void MakeColumnVector(index_t col, GenVector<T> *dest) const {
    DEBUG_BOUNDS(col, n_cols_);
    dest->Alias(n_rows_ * col + ptr_, n_rows_);
  }
  
  /**
   * Makes an alias of a subvector of particular column.
   *
   * @param col the column to alias
   * @param start_row the first row to put in the subvector
   * @param n_rows_new the number of rows of the subvector
   * @param dest a pointer to an uninitialized vector, which will be
   *        initialized as an alias to the particular column's subvector
   */
  void MakeColumnSubvector(index_t col, index_t start_row, index_t n_rows_new,
      GenVector<T> *dest) const {
    DEBUG_BOUNDS(col, n_cols_);
    DEBUG_BOUNDS(start_row, n_rows_);
    DEBUG_BOUNDS(start_row + n_rows_new, n_rows_ + 1);
    dest->Alias(n_rows_ * col + start_row + ptr_, n_rows_new);
  }
  
  /**
   * Retrieves a pointer to a contiguous array corresponding to a particular
   * column.
   *
   * @param col the column number
   * @return an array where the i'th element is the i'th row of that
   *         particular column
   */
  T *GetColumnPtr(index_t col) {
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
  const T *GetColumnPtr(index_t col) const {
    DEBUG_BOUNDS(col, n_cols_);
    return n_rows_ * col + ptr_;
  }
  
  /**
   * Copies a vector to a matrix column.
   * @param col1 the column number of this matrix
   * @param col2 the column number of the other matrix
   * @param mat the other matrix
   * @return nothing
   */  
   void CopyColumnFromMat(index_t col1, index_t col2, GenMatrix<T> &mat) {
     DEBUG_BOUNDS(col1, n_cols_);
     DEBUG_BOUNDS(col2, mat.n_cols());
     DEBUG_ASSERT(n_rows_==mat.n_rows());
     memcpy(ptr_ + n_rows_ * col1, mat.GetColumnPtr(col2), n_rows_*sizeof(T));
   }
  /**
   * Copies a block of columns to a matrix column.
   * @param col1 the column number of this matrix
   * @param col2 the column number of the other matrixa
   * @param ncols the number of columns
   * @param mat the other matrix
   * @return nothing
   */  
   void CopyColumnFromMat(index_t col1, index_t col2, index_t ncols, GenMatrix<T> &mat) {
     DEBUG_BOUNDS(col1, n_cols_);
     DEBUG_BOUNDS(col2, mat.n_cols());
     DEBUG_BOUNDS(col1+ncols-1, n_cols_);
     DEBUG_BOUNDS(col2+ncols-1, mat.n_cols());
     DEBUG_ASSERT(n_rows_==mat.n_rows());
     memcpy(ptr_ + n_rows_ * col1, mat.GetColumnPtr(col2), ncols*n_rows_*sizeof(T));
   }

   /**
   * Copies a column of matrix 1  to a column of matrix 2.
   * @param col1 the column number
   * @return nothing
   */  
   void CopyVectorToColumn(index_t col, GenVector<T> &vec) {
     DEBUG_BOUNDS(col, n_cols_);
     memcpy(ptr_ + n_rows_ * col, vec.ptr(), n_rows_*sizeof(T));
   }
 
  /**
   * Changes the number of columns, but REQUIRES that there are no aliases
   * to this matrix anywhere else.
   *
   * If the size is increased, the remaining space is not initialized.
   *
   * @param new_n_cols the new number of columns
   */
  void ResizeNoalias(index_t new_n_cols) {
    DEBUG_ASSERT(should_free_); // the best assert we can do
    n_cols_ = new_n_cols;
    ptr_ = mem::Realloc(ptr_, n_elements());
  }
  
  /**
   * Swaps all values in this matrix with values in the other.
   *
   * This is different from Swap, because Swap will only change what these
   * point to.
   *
   * @param other an identically sized vector to swap values with
   */
  void SwapValues(GenMatrix* other) {
    DEBUG_ASSERT(n_cols() == other->n_cols());
    DEBUG_ASSERT(n_rows() == other->n_rows());
    mem::Swap(ptr_, other->ptr_, n_elements());
  }
  
  /**
   * Copies the values from another matrix to this matrix.
   *
   * @param other the vector to copy from
   */
  void CopyValues(const GenMatrix& other) {
    DEBUG_ASSERT(n_rows() == other.n_rows());
    DEBUG_ASSERT(n_cols() == other.n_cols());
    mem::Copy(ptr_, other.ptr_, n_elements());
  }

  /**
   * Prints to a stream as a debug message.
   *
   * @param name a name that will be printed with the matrix
   * @param stream the stream to print to, defaults to @c stderr
   */
  void PrintDebug(const char *name = "", FILE *stream = stderr) const {
    fprintf(stream, "----- MATRIX %s ------\n", name);
    for (index_t r = 0; r < n_rows(); r++) {
      for (index_t c = 0; c < n_cols(); c++) {
        fprintf(stream, "%+3.3f ", get(r, c));
      }
      fprintf(stream, "\n");
    }
  }
  
 public:
  /**
   * Returns a pointer to the very beginning of the matrix, stored
   * in a column-major format.
   *
   * This is suitable for BLAS and LAPACK calls.
   */
  const T *ptr() const {
    return ptr_;
  }
  
  /**
   * Returns a pointer to the very beginning of the matrix, stored
   * in a column-major format.
   *
   * This is suitable for BLAS and LAPACK calls.
   */
  T *ptr() {
    return ptr_;
  }
  
  /**
   * Gets a particular double at the specified row and column.
   *
   * @param r the row number
   * @param c the column number
   */
  T get(index_t r, index_t c) const {
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
  void set(index_t r, index_t c, T v) {
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
  T &ref(index_t r, index_t c) {
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
    DEBUG_ASSERT_MSG(n_rows_ == BIG_BAD_NUMBER, "Cannot re-init matrices.");
  }
  
  void Uninitialize_() {
    DEBUG_POISON_PTR(ptr_);
    DEBUG_ONLY(n_rows_ = BIG_BAD_NUMBER);
    DEBUG_ONLY(n_cols_ = BIG_BAD_NUMBER);
  } 

};

/**
 * Low-overhead vector if length is known at compile time.
 */
template<int t_length>
class SmallVector : public Vector {
 private:
  double array_[t_length];
  
 public:
  SmallVector() {
    Alias(array_, t_length);
  }
  ~SmallVector() {}
  
 public:
  index_t length() const {
    return t_length;
  }

  double *ptr() {
    return array_;
  }
  
  const double *ptr() const {
    return array_;
  }
  
  double operator [] (index_t i) const {
    DEBUG_BOUNDS(i, t_length);
    return array_[i];
  }
  
  double &operator [] (index_t i) {
    DEBUG_BOUNDS(i, t_length);
    return array_[i];
  }
  
  double get(index_t i) const {
    DEBUG_BOUNDS(i, t_length);
    return array_[i];
  }
};

/** @brief A Matrix is a GenMatrix of double's.
 */
typedef GenMatrix<double> Matrix;

/**
 * Low-overhead matrix if size is known at compile time.
 */
template<int t_rows, int t_cols>
class SmallMatrix : public Matrix {
 private:
  double array_[t_cols][t_rows];

 public:
  SmallMatrix() {
    Alias(array_[0], t_rows, t_cols);
  }
  ~SmallMatrix() {}

 public:
  const double *ptr() const {
    return array_[0];
  }

  double *ptr() {
    return array_[0];
  }

  double get(index_t r, index_t c) const {
    DEBUG_BOUNDS(r, t_rows);
    DEBUG_BOUNDS(c, t_cols);
    return array_[c][r];
  }

  void set(index_t r, index_t c, double v) {
    DEBUG_BOUNDS(r, t_rows);
    DEBUG_BOUNDS(c, t_cols);
    array_[c][r] = v;
  }

  double &ref(index_t r, index_t c) {
    DEBUG_BOUNDS(r, t_rows);
    DEBUG_BOUNDS(c, t_cols);
    return array_[c][r];
  }

  index_t n_cols() const {
    return t_cols;
  }

  index_t n_rows() const {
    return t_rows;
  }

  size_t n_elements() const {
    // TODO: putting the size_t on the outside may be faster (32-bit
    // versus 64-bit multiplication in cases) but is more likely to result
    // in bugs
    return size_t(t_rows) * size_t(t_cols);
  }

  double *GetColumnPtr(index_t col) {
    DEBUG_BOUNDS(col, t_cols);
    return array_[col];
  }

  const double *GetColumnPtr(index_t col) const {
    DEBUG_BOUNDS(col, t_cols);
    return array_[col];
  }
};

#endif

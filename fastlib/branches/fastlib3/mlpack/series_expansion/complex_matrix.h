#ifndef COMPLEX_MATRIX_H
#define COMPLEX_MATRIX_H

#include "fastlib/fastlib.h"
#include <complex>

template<typename T>
class ComplexVector {

 private:

  /** Linearized matrix (column-major). */
  T *ptr_;

  /** Number of elements. */
  index_t length_;

  /** Whether I am a strong copy (not an alias). */
  bool should_free_;

  OBJECT_TRAVERSAL_ONLY(ComplexVector) {
    OT_OBJ(length_);
    OT_ALLOC(ptr_, length_);
  }
  OT_REFILL_TRANSIENTS(ComplexVector) {
    should_free_ = true;
  }

 public:

  /**
   * Creates a completely uninitialized Vector which must be initialized.
   */
  ComplexVector() {
    DEBUG_ONLY(Uninitialize_());
  }
  
  /**
   * Copy constructor -- for use in collections.
   */
  ComplexVector(const ComplexVector& other) {
    DEBUG_ONLY(Uninitialize_());
    Copy(other);
  }
  ASSIGN_VIA_COPY_CONSTRUCTION(ComplexVector);
  
  /**
   * Destroys the Vector, freeing the memory if this copy is not an alias.
   */
  ~ComplexVector() {
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
      mem::DebugPoison(ptr_, 2 * length_);
      mem::Free(ptr_);
    }
    
    DEBUG_ONLY(Uninitialize_());
  }

  /**
   * Creates a vector of a particular length, but does not initialize the
   * values in it.
   */
  void Init(index_t in_length) {
    ptr_ = mem::Alloc<T>(2 * in_length);
    length_ = in_length;
    should_free_ = true;
  }

  /**
   * Sets all elements to the same value.
   */
  void SetAll(T d) {
    mem::RepeatConstruct(ptr_, d, 2 * length_);
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
  void Copy(const ComplexVector<T> & other) {
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
    
    ptr_ = mem::AllocCopy(doubles, 2 * in_length);
    length_ = in_length;
    should_free_ = true;
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
  void WeakCopy(const ComplexVector<T> & other) {
    Alias(other);
  }
  
  /**
   * Makes this vector an alias of another vector.
   *
   * @param other the other vector
   */
  void Alias(const ComplexVector& other) {
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
  void Own(ComplexVector* other) {
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
   * Copies the values from another vector to this vector.
   *
   * @param other the vector to copy from
   */
  void CopyValues(const ComplexVector& other) {
    DEBUG_ASSERT(length() == other.length());
    mem::Copy(ptr_, other.ptr_, 2 * length_);
  }

  /**
   * Copies all of the values from an array of doubles to this vector.
   *
   * @param src_ptr the vector to copy from, must have at least
   *        length() elements
   */
  void CopyValues(const T *src_ptr) {
    mem::Copy(ptr_, src_ptr, 2 * length_);
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
    return ptr_[2 * i];
  }
  
  /**
   * Gets a mutable reference to the i'th element of this vector.
   */
  T &operator [] (index_t i) {
    DEBUG_BOUNDS(i, length_);
    return ptr_[2 * i];
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
  std::complex<T> get(index_t i) const {
    DEBUG_BOUNDS(i, length_);
    std::complex<T> result;
    result.real() = ptr_[2 * i];
    result.imag() = ptr_[2 * i + 1];

    return result;
  }

  void set(index_t i, std::complex<T> new_value) {
    DEBUG_BOUNDS(i, length_);
    ptr_[2 * i] = new_value.real();
    ptr_[2 * i + 1] = new_value.imag();
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
      std::complex<T> number = get(i);
      fprintf(stream, "%g + %g * i\n", number.real(), number.imag());
    }
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

class ComplexMatrix {

 private:

  /** Linearized matrix (column-major). */
  double *ptr_;

  /** Number of rows. */
  index_t n_rows_;

  /** Number of columns. */
  index_t n_cols_;

  /** Whether I am a strong copy (not an alias). */
  bool should_free_;

  OBJECT_TRAVERSAL_ONLY(ComplexMatrix) {
    OT_OBJ(n_rows_);
    OT_OBJ(n_cols_);
    OT_ALLOC(ptr_, n_elements());
  }
  OT_REFILL_TRANSIENTS(ComplexMatrix) {
    should_free_ = false;
  }

 public:

  /**
   * Creates a Matrix with uninitialized elements of the specified size.
   */
  ComplexMatrix(index_t in_rows, index_t in_cols) {
    DEBUG_ONLY(Uninitialize_());
    Init(in_rows, in_cols);
  }

  /**
   * Copy constructor -- for use in collections.
   */
  ComplexMatrix(const ComplexMatrix& other) {
    DEBUG_ONLY(Uninitialize_());
    Copy(other);
  }
  ASSIGN_VIA_COPY_CONSTRUCTION(ComplexMatrix);

  /**
   * Creates a matrix that can be initialized.
   */
  ComplexMatrix() {
    DEBUG_ONLY(Uninitialize_());
  }

  /**
   * Empty destructor.
   */
  ~ComplexMatrix() {
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
      mem::DebugPoison(ptr_, 2 * n_rows_ * n_cols_);
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
    ptr_ = mem::Alloc<double>(2 * in_rows * in_cols);
    n_rows_ = in_rows;
    n_cols_ = in_cols;
    should_free_ = true;
  }

  /**
   * Sets the entire matrix to zero.
   */
  void SetAll(double d) {
    mem::RepeatConstruct(ptr_, d, 2 * n_elements());
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
  void Copy(const ComplexMatrix& other) {
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
    
    ptr_ = mem::AllocCopy(ptr_in, 2 * n_rows_in * n_cols_in);
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
  void Alias(const ComplexMatrix& other) {
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
  void WeakCopy(const ComplexMatrix& other) {
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
  void Own(ComplexMatrix* other) {
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
  void Own(double *ptr_in, index_t n_rows_in, index_t n_cols_in) {
    DEBUG_ONLY(AssertUninitialized_());
    
    ptr_ = ptr_in;
    n_rows_ = n_rows_in;
    n_cols_ = n_cols_in;
    should_free_ = true;
  }
  
  /**
   * Copies the values from another matrix to this matrix.
   *
   * @param other the vector to copy from
   */
  void CopyValues(const ComplexMatrix& other) {
    DEBUG_ASSERT(n_rows() == other.n_rows());
    DEBUG_ASSERT(n_cols() == other.n_cols());
    mem::Copy(ptr_, other.ptr_, 2 * n_elements());
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
        fprintf(stream, "(%+3.3f, %+3.3f) ", get(r, c).real(),
		get(r, c).imag());
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
  std::complex<double> get(index_t r, index_t c) const {
    DEBUG_BOUNDS(r, n_rows_);
    DEBUG_BOUNDS(c, n_cols_);

    std::complex<double> result;
    result.real() = ptr_[2 * (c * n_rows_ + r)];
    result.imag() = ptr_[2 * (c * n_rows_ + r) + 1];
    return result;
  }
 
  /**
   * Sets the value at the row and column.
   *
   * @param r the row number
   * @param c the column number
   * @param v the value to set
   */ 
  void set(index_t r, index_t c, std::complex<double> v) {
    DEBUG_BOUNDS(r, n_rows_);
    DEBUG_BOUNDS(c, n_cols_);
    ptr_[2 * (c * n_rows_ + r)] = v.real();
    ptr_[2 * (c * n_rows_ + r) + 1] = v.imag();
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

#endif

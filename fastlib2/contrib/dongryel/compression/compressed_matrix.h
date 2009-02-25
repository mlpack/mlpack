// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file compressed_matrix.h
 *
 * Basic double-precision vector and matrix classes.
 */

#ifndef COMPRESSED_MATRIX_H
#define COMPRESSED_MATRIX_H

#include "fastlib/base/base.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "zlib.h"

template<typename T>
class CompressedVector {

 private:
  
  /** The pointer to the array of chars. */
  Bytef *ptr_;
  
  /** The number of compressed elements. */
  index_t internal_length_;

  /** The length of the array. */
  index_t length_;
  
 public:

  /**
   * Creates a completely uninitialized Vector which must be initialized.
   */
  CompressedVector() {
    ptr_ = NULL;
    internal_length_ = 0;
    length_ = 0;
  }
  
  /**
   * Destroys the Vector, freeing the memory if this copy is not an alias.
   */
  ~CompressedVector() {
    Destruct();
  }
  
  /**
   * Uninitializes so that you can call another initializer.
   */
  void Destruct() {
    mem::DebugPoison(ptr_, internal_length_);
    mem::Free(ptr_);
  }

  /** @brief Creates a compressed vector from uncompressed vector.
   */
  void Init(const GenVector<T> &uncompressed_vector) {

    index_t uncompressed_size = uncompressed_vector.length() * sizeof(T);
    uLongf buffer_size = compressBound(uncompressed_size);
    Bytef *tmp_buffer = mem::Alloc<Bytef>(buffer_size);
    
    int success_flag = compress(tmp_buffer, &buffer_size, 
				(const Bytef *) uncompressed_vector.ptr(), 
				uncompressed_vector.length() * sizeof(T));

    if(success_flag == Z_MEM_ERROR) {
      NOTIFY("There was not enough memory for compressing.");
    }
    else if(success_flag == Z_BUF_ERROR) {
      NOTIFY("There was not enough room in the buffer.");
    }
    else {
      NOTIFY("Compression is a success: %d bytes into %d bytes!",
	     uncompressed_size, buffer_size);
    }
    
    // Reallocate the new memory block.
    ptr_ = mem::Alloc<Bytef>(buffer_size);
    memcpy(ptr_, tmp_buffer, buffer_size);
    internal_length_ = buffer_size;
    mem::Free(tmp_buffer);    
  }
  
 public:

  /** The number of elements in this vector. */
  index_t length() const {
    return length_;
  }
  
  /**
   * Gets the i'th element of this vector.
   */
  T operator [] (index_t i) const {
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

};

#endif

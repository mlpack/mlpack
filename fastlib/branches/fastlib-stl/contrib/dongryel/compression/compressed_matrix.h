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

template<typename T, int block_size = 100>
class CompressedVector {

 private:
  
  /** The pointer to the array of chars. */
  Bytef *ptr_;
  
  /** The number of bytes used for compression. */
  index_t internal_length_;

  /** @brief The offsets required to access each block of elements in
   *         the compressed sream.
   */
  int *index_offsets_;

  int num_blocks_;

  /** The number of elements in the vector. */
  index_t length_;

  /** The currently decompressed block number.
   */
  index_t current_decompressed_block_;

  /** @brief The cache of decompressed block of elements.
   */
  T *decompressed_cache_;

 public:

  /**
   * Creates a completely uninitialized Vector which must be initialized.
   */
  CompressedVector() {
    ptr_ = NULL;
    index_offsets_ = NULL;
    internal_length_ = 0;
    length_ = 0;
    current_decompressed_block_ = -1;
    decompressed_cache_ = NULL;
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
    mem::Free(index_offsets_);
    mem::Free(ptr_);
    mem::Free(decompressed_cache_);
  }

  /** @brief Creates a compressed vector from uncompressed vector.
   */
  void Copy(const GenVector<T> &uncompressed_vector) {

    index_t uncompressed_block_size = block_size * sizeof(T);
    num_blocks_ = (int) 
      ceilf(((double)uncompressed_vector.length()) / ((double) block_size));

    // Allocate the index offset array.
    index_offsets_ = mem::Alloc<int>(num_blocks_);

    // Allocate the cache for the decompressed block of elements.
    decompressed_cache_ = mem::Alloc<T>(block_size);

    uLongf buffer_size = num_blocks_ * compressBound(uncompressed_block_size);
    Bytef *tmp_buffer = mem::Alloc<Bytef>(buffer_size);

    // Iterate over each block of elements and compress, marking the
    // starting byte for each.
    uLongf remaining_buffer_size = buffer_size;
    uLongf total_bytes_chewed = 0;
    for(index_t i = 0; i < num_blocks_; i++) {
      int success_flag = compress(tmp_buffer + total_bytes_chewed, 
				  &remaining_buffer_size, 
				  (const Bytef *) uncompressed_vector.ptr() +
				  i * block_size * sizeof(T), 
				  block_size * sizeof(T));
      
      if(success_flag == Z_MEM_ERROR) {
	//NOTIFY("There was not enough memory for compressing.");
      }
      else if(success_flag == Z_BUF_ERROR) {
	//NOTIFY("There was not enough room in the buffer.");
      }
      else {
	//NOTIFY("Compression is a success: %d bytes into %d bytes!",
	//     uncompressed_block_size, (int) remaining_buffer_size);
      }
      index_offsets_[i] = total_bytes_chewed;
      total_bytes_chewed += remaining_buffer_size;
      remaining_buffer_size = buffer_size - total_bytes_chewed;
    }
    
    // Reallocate the new memory block.
    ptr_ = mem::Alloc<Bytef>(total_bytes_chewed);
    memcpy(ptr_, tmp_buffer, total_bytes_chewed);
    internal_length_ = total_bytes_chewed;
    mem::Free(tmp_buffer);    

    // Set the number of elements.
    length_ = uncompressed_vector.length();
  }

  /** The number of elements in this vector. */
  index_t length() const {
    return length_;
  }
  
  /**
   * Gets the i'th element of this vector.
   */
  T operator [] (index_t i) {
    DEBUG_BOUNDS(i, length_);

    // Compute the block that contains this element.
    index_t block_index = i / block_size;

    // If the current block is not in the cache, then fetch the block.
    if(block_index != current_decompressed_block_) {
      uLongf num_bytes_written = block_size * sizeof(T);
      int num_bytes_to_decompress = (block_index == num_blocks_ - 1) ?
	(internal_length_ - index_offsets_[block_index]):
	(index_offsets_[block_index + 1] - index_offsets_[block_index]);
      uncompress((Bytef *) decompressed_cache_, &num_bytes_written,
		 ptr_ + index_offsets_[block_index], num_bytes_to_decompress);
      current_decompressed_block_ = block_index;
    }
    
    return decompressed_cache_[i % block_size];
  }
};

#endif

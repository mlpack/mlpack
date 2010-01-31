/**
 * @file intmap.h
 *
 * Dense integer-to-value map.
 */

#ifndef COL_INTMAP_H
#define COL_INTMAP_H

#include "fastlib/base/base.h"

/**
 * A dense grow-as-needed array that serves as an integer-keyed map.
 */
template<class TValue>
class DenseIntMap {
 public:
  typedef TValue Value;

 private:
  Value *ptr_;
  index_t size_;
  Value default_value_;

  OT_DEF(DenseIntMap) {
    OT_MY_OBJECT(size_);
    OT_MY_OBJECT(default_value_);
    OT_MALLOC_ARRAY(ptr_, size_);
  }

 public:
  /** Creates a blank mapping. */
  void Init() {
    ptr_ = NULL;
    size_ = 0;
  }

  /** Accesses the default value.  Use this to set it. */
  Value& default_value() {
    return default_value_;
  }
  /** Accesses the default value. */
  const Value& default_value() const {
    return default_value_;
  }

  /**
   * Gets a non-inclusive upper bound on the last non-default element.
   *
   * Iterating up to size() is guaranteed to hit all elements without
   * growing the internal array.
   */
  index_t size() const {
    return size_;
  }

  /**
   * Accesses an element, expanding if necessary.
   *
   * If you are just probing, beware that this might actually grow the array!
   */
  Value& operator [] (index_t index) {
    DEBUG_BOUNDS(index, BIG_BAD_NUMBER);
    if (unlikely(index >= size_)) {
      index_t old_size = size_;
      size_ = std::max(size_ * 2, index + 1);
      ptr_ = mem::Realloc(ptr_, size_);
      for (index_t i = old_size; i < size_; i++) {
        new(ptr_+i)Value(default_value_);
      }
    }
    return ptr_[index];
  }
  /**
   * Accesses an element from a static context.
   */
  const Value& operator [] (index_t index) const {
    return get(index);
  }
  /**
   * Accesses an element, never growing the internal representation.
   */
  const Value& get(index_t index) const {
    DEBUG_BOUNDS(index, BIG_BAD_NUMBER);
    if (likely(index < size_)) {
      return ptr_[index];
    } else {
      return default_value_;
    }
  }
};

#endif

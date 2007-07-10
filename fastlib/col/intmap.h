/**
 * @file intmap.h
 *
 * Dense integer-to-value map.
 */

template<class TValue>
class DenseIntMap {
  FORBID_COPY(DenseIntMap);

 public:
  typedef TValue Value;

 private:
  Value *ptr_;
  index_t size_;
  Value default_value_;

 public:
  DenseIntMap() {
    DEBUG_POISON_PTR(ptr_);
    size_ = BIG_BAD_NUMBER;
  }
  ~DenseIntMap() {
    DEBUG_ASSERT(size_ != BIG_BAD_NUMBER);
    mem::Free(ptr_);
  }

  void Init() {
    ptr_ = NULL;
    size_ = 0;
  }

  Value& default_value() {
    return default_value_;
  }
  const Value& default_value() const {
    return default_value_;
  }

  Value& operator [] (index_t index) {
    if (unlikely(index >= size_)) {
      index_t old_size = size_;
      size_ = max(size_ * 2, index + 1);
      ptr_ = mem::Resize(ptr_, size_);
      for (index_t i = old_size; i < size_; i++) {
        new(ptr_+i)Value(default_value_);
      }
    }
    return ptr_[index];
  }
  const Value& operator [] (index_t index) const {
    return get(index);
  }
  const Value& get(index_t index) const {
    if (likely(index < size_)) {
      return ptr_[index];
    } else {
      return default_value_;
    }
  }
};

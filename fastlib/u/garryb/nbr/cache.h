#ifndef NBR_CACHE_H
#define NBR_CACHE_H

#include "blockdev.h"

/**
 * Extra-simple cache for when everything fits in RAM.
 *
 * All methods here must be locked by a mutex.
 */
class SmallCache : public BlockDeviceWrapper {
  FORBID_COPY(SmallCache);

 private:
  struct Metadata {
    Metadata() : data(NULL), lock_count(0) {}
    /** current version of the data, can be NULL */
    char *data;
    char is_dirty;
    char lock_count;
  };

 private:
  Mutex mutex_;
  ArrayList<Metadata> metadatas_;
  BlockActionHandler *handler_;
  mode_t mode_;

 public:
  SmallCache() {}
  virtual ~SmallCache();

  mode_t mode() const {
    return mode_;
  }

  void Init(BlockDevice *inner_in, BlockActionHandler *handler_in,
     mode_t mode_in);

  char *StartRead(blockid_t blockid);
  char *StartWrite(blockid_t blockid);
  void StopRead(blockid_t blockid);
  void StopWrite(blockid_t blockid);
  void Flush(blockid_t begin_block, offset_t begin_offset,
    blockid_t last_block, offset_t end_offset);
  void Close();

  virtual void Read(blockid_t blockid,
      offset_t begin, offset_t end, char *data);
  virtual void Write(blockid_t blockid,
      offset_t begin, offset_t end, const char *data);

  virtual blockid_t AllocBlock() {
    blockid_t blockid = BlockDeviceWrapper::AllocBlock();
    metadatas_.Resize(n_blocks());
    return blockid;
  }

 private:
  void PerformCacheMiss_(blockid_t blockid);
  void Writeback_(blockid_t blockid, offset_t begin, offset_t end);

  Metadata *GetBlock_(blockid_t blockid) {
    Metadata *metadata = &metadatas_[blockid];
    char *data = metadata->data;

    if (unlikely(data == NULL)) {
      PerformCacheMiss_(blockid);
    }

    return metadata;
  }
};

template<typename T>
class CacheArrayBlockActionHandler : public BlockActionHandler {
  FORBID_COPY(CacheArrayBlockActionHandler);

 private:
  char *default_elem_;
  size_t n_elem_bytes_;

 public:
  CacheArrayBlockActionHandler() {
    n_elem_bytes_ = BIG_BAD_NUMBER;
    DEBUG_POISON_PTR(default_elem_);
  }

  ~CacheArrayBlockActionHandler() {
    mem::Free(default_elem_);
    n_elem_bytes_ = BIG_BAD_NUMBER;
    DEBUG_POISON_PTR(default_elem_);
  }

  void Init(const T& default_obj) {
    n_elem_bytes_ = ot::PointerFrozenSize(default_obj);
    default_elem_ = mem::Alloc<char>(n_elem_bytes_);
    ot::PointerFreeze(default_obj, default_elem_);
  }

  void BlockInitFrozen(size_t bytes, char *block) {
    index_t elems = bytes / n_elem_bytes_;
    for (index_t i = 0; i < elems; i++) {
      mem::CopyBytes(block, default_elem_, n_elem_bytes_);
      block += n_elem_bytes_;
    }
  }

  void BlockRefreeze(size_t bytes, const char *old_location, char *block) {
    index_t elems = bytes / n_elem_bytes_;
    for (index_t i = 0; i < elems; i++) {
      ot::PointerRefreeze(reinterpret_cast<const T*>(old_location), block);
      block += n_elem_bytes_;
      old_location += n_elem_bytes_;
    }
  }

  void BlockThaw(size_t bytes, char *block) {
    index_t elems = bytes / n_elem_bytes_;
    for (index_t i = 0; i < elems; i++) {
      ot::PointerThaw<T>(block);
      block += n_elem_bytes_;
    }
  }
  
  size_t n_elem_bytes() {
    return n_elem_bytes_;
  }
};

// LIMITATION: This type of cache array assumes that everything fits in
// memory (it never releases locks).
template<typename T>
class CacheArray {
  FORBID_COPY(CacheArray);

 public:
  typedef T Element;

 private:
  struct Metadata {
    Metadata() : data(NULL) {
      DEBUG_ONLY(lock_count = 0);
    }
    char *data;
#ifdef DEBUG
    // lock count will be useful when we start using a FIFO layer
    char lock_count;
#endif
  };

 private:
  unsigned int n_block_elems_log_;
  unsigned int n_block_elems_mask_;
  ArrayList<Metadata> metadatas_;
  unsigned int n_elem_bytes_;
  
  index_t begin_;
  index_t end_;
  unsigned int n_block_elems_;
  BlockDevice::blockid_t begin_block_;
  BlockDevice::blockid_t end_block_;
  mode_t mode_;
  
  SmallCache *cache_;

 public:
  CacheArray() {}
  virtual ~CacheArray() {}

  /** Reopens another cache array */
  void Init(CacheArray *other, BlockDevice::mode_t mode_in) {
    Init(other, mode_in, other->begin_index(), other->end_index());
  }

  /** Reopens another cache array */
  void Init(CacheArray *other, BlockDevice::mode_t mode_in,
      index_t begin_index_in, index_t end_index_in) {
    Init(other->cache_, mode_in, begin_index_in, end_index_in,
        other->n_block_elems_, other->n_elem_bytes_);
  }

  void Init(SmallCache *cache_in, BlockDevice::mode_t mode_in,
      index_t begin_index_in, index_t end_index_in,
      index_t n_block_elems_in, size_t n_elem_bytes_in);

  index_t begin_index() const {
    return begin_;
  }
  index_t end_index() const {
    return end_;
  }
  
  unsigned int n_elem_bytes() const {
    return n_elem_bytes_;
  }

  unsigned int n_block_elems() const {
    return n_block_elems_;
  }
  
  const SmallCache *cache() const {
    return cache_;
  }

  const Element *StartRead(index_t element_id) {
    return CheckoutElement_(element_id);
  }

  Element *StartWrite(index_t element_id) {
    DEBUG_ASSERT(mode_ != BlockDevice::READ);
    return CheckoutElement_(element_id);
  }

  void StopRead(index_t element_id) {
    DEBUG_ONLY(BoundsCheck_(element_id));
    ReleaseElement_(element_id);
  }

  void StopWrite(index_t element_id) {
    DEBUG_ONLY(BoundsCheck_(element_id));
    DEBUG_ASSERT(mode_ != BlockDevice::READ);
    ReleaseElement_(element_id);
  }

  void Swap(index_t index_a, index_t index_b) {
    DEBUG_ONLY(BoundsCheck_(index_a));
    DEBUG_ONLY(BoundsCheck_(index_b));
    DEBUG_ASSERT(mode_ != BlockDevice::READ);
    char *a = reinterpret_cast<char*>(CheckoutElement_(index_a));
    char *b = reinterpret_cast<char*>(CheckoutElement_(index_b));
    mem::Swap(a, b, n_elem_bytes_);
    ReleaseElement_(index_a);
    ReleaseElement_(index_b);
  }

  void Copy(index_t index_src, index_t index_dest) {
    DEBUG_ONLY(BoundsCheck_(index_src));
    DEBUG_ONLY(BoundsCheck_(index_dest));
    DEBUG_ASSERT(mode_ != BlockDevice::READ);
    const char *src = reinterpret_cast<char*>(CheckoutElement_(index_src));
    char *dest = reinterpret_cast<char*>(CheckoutElement_(index_dest));
    mem::Copy(dest, src, n_elem_bytes_);
    ReleaseElement_(index_src);
    ReleaseElement_(index_dest);
  }

  /**
   * Flushes all changes.
   */
  void Flush();
  
  index_t Alloc() {
    end_++;
    BlockDevice::blockid_t block_computed = 
        (end_ + n_block_elems_ - 1) >> n_block_elems_log_;
    if (block_computed != end_block_) {
      end_block_ = cache_->AllocBlock();
      metadatas_.Resize(end_block_ - begin_block_);
      // Okay, notify the lower layers we're allocating.
      DEBUG_ASSERT_MSG(block_computed == end_block_,
          "Distributed data structure creation "
          "is not yet supported by CacheArray.");
    }
    return end_ - 1;
  }

 private:
  void BoundsCheck_(index_t element_id) {
    DEBUG_ASSERT(element_id >= begin_);
    DEBUG_ASSERT(element_id < end_);
  }

  COMPILER_NOINLINE
  Element *HandleCacheMiss_(index_t element_id);

  Element *CheckoutElement_(index_t element_id) {
    DEBUG_ONLY(BoundsCheck_(element_id));

    BlockDevice::offset_t offset =
        (element_id & n_block_elems_mask_) * n_elem_bytes_;
    BlockDevice::blockid_t blockid = element_id >> n_block_elems_log_;

    Metadata *metadata = &metadatas_[blockid - begin_block_];

    DEBUG_ONLY(metadata->lock_count++);

    if (unlikely(!metadata->data)) {
      return HandleCacheMiss_(element_id);
    } else {
      Element *ptr = reinterpret_cast<Element*>(metadata->data + offset);
      return ptr;
    }
  }
  
  void ReleaseElement_(index_t element_id) {
    DEBUG_ONLY(BoundsCheck_(element_id));
    DEBUG_ONLY(--metadatas_[
        (element_id >> n_block_elems_log_) - begin_block_].lock_count);
  }
};

template<typename T>
void CacheArray<T>::Init(SmallCache *cache_in, BlockDevice::mode_t mode_in,
    index_t begin_index_in, index_t end_index_in,
    index_t n_block_elems_in, size_t n_elem_bytes_in) {
  cache_ = cache_in;
  begin_ = begin_index_in;
  end_ = end_index_in;
  n_block_elems_ = n_block_elems_in;
  // Cache size must be a power of 2.
  n_block_elems_log_ = math::IntLog2(n_block_elems_);
  n_block_elems_mask_ = n_block_elems_ - 1;
  n_elem_bytes_ = n_elem_bytes_in;
  
  begin_block_ = begin_ / n_block_elems_;
  end_block_ = (end_ + n_block_elems_ - 1) / n_block_elems_;
  
  metadatas_.Init(end_block_ - begin_block_);
  
  mode_ = mode_in;
}

template<typename T>
void CacheArray<T>::Flush() {
  for (BlockDevice::blockid_t blockid = begin_block_;
      blockid < end_block_; blockid++) {
    Metadata *metadata = &metadatas_[blockid - begin_block_];
    if (metadata->data != NULL) {
      if (mode_ != BlockDevice::READ) {
        cache_->StopWrite(blockid);
      } else {
        cache_->StopRead(blockid);
      }
      DEBUG_SAME_INT(metadata->lock_count, 0);
      metadata->data = NULL;
    }
  }
  cache_->Flush(
      begin_ / n_block_elems_, (begin_ & (n_block_elems_ - 1)) * n_elem_bytes_,
      end_ / n_block_elems_, (end_ & (n_block_elems_ - 1)) * n_elem_bytes_);
}

template<typename T>
typename CacheArray<T>::Element* CacheArray<T>::HandleCacheMiss_(
    index_t element_id) {
  BlockDevice::blockid_t blockid = element_id >> n_block_elems_log_;
  Metadata *metadata = &metadatas_[blockid];
  
  if (mode_ != BlockDevice::READ) {
    metadata->data = cache_->StartWrite(blockid);
  } else {
    metadata->data = cache_->StartRead(blockid);
  }

  BlockDevice::offset_t offset =
      uint(element_id & (n_block_elems_ - 1)) * n_elem_bytes_;
  
  return reinterpret_cast<Element*>(metadata->data + offset);
}

template<typename T>
class TempCacheArray : public CacheArray<T> {
 private:
  SmallCache underlying_cache_;
  NullBlockDevice null_device_;
  
 public:
  virtual ~TempCacheArray() {
    CacheArray<T>::Flush();
  }

  /** Creates a blank, temporary cached array */
  void Init(const T& default_obj,
      index_t n_elems_in,
      unsigned int n_block_elems_in) {
    CacheArrayBlockActionHandler<T> *handler =
        new CacheArrayBlockActionHandler<T>;
    handler->Init(default_obj);

    null_device_.Init((n_elems_in + n_block_elems_in + 1) / n_block_elems_in,
        n_block_elems_in * handler->n_elem_bytes());
    underlying_cache_.Init(&null_device_, handler, BlockDevice::TEMP);
    
    CacheArray<T>::Init(&underlying_cache_, BlockDevice::TEMP,
        0, n_elems_in, n_block_elems_in, handler->n_elem_bytes());
  }
};

#endif

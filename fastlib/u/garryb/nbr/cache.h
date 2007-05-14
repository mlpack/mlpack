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
    Metadata() : data(NULL), is_dirty(0) {}
    /** current version of the data, can be NULL */
    char *data;
    char lock_count;
    char is_dirty;
  };

 private:
  Mutex mutex_;
  ArrayList<Metadata> metadata_;
  BlockActionHandler *handler_;

 public:
  virtual ~SmallCache();

  char *StartRead(blockid_t blockid);
  char *StartWrite(blockid_t blockid);
  void StopRead(blockid_t blockid);
  void StopWrite(blockid_t blockid);

  void Init(BlockDevice *inner_in, BlockActionHandler *handler_in) {
    BlockDeviceWrapper::Init(inner_in);
    handler_ = handler_in;
  }

  virtual void Read(blockid_t blockid, char *data) {
    abort();
  }

  virtual void Write(blockid_t blockid, const char *data) {
    abort();
  }

  virtual blockid_t AllocBlock() {
    blockid_t blockid = BlockDeviceWrapper::AllocBlock();
    metadata_.Resize(n_blocks());
    return blockid;
  }

 private:
  void PerformCacheMiss_(blockid_t blockid);

  Metadata *GetBlock_(blockid_t blockid) {
    Metadata *metadata = &metadata_[blockid];
    char *data = metadata->data;

    if (unlikely(data == NULL)) {
      PerformCacheMiss_(blockid);
    }

    return metadata;
  }
};

SmallCache::~SmallCache() {
  for (index_t i = 0; i < metadata_.size(); i++) {
    Metadata *metadata = &metadata_[i];
    mem::Free(metadata->data);
    DEBUG_ASSERT(metadata_[i].lock_count == 0);
    DEBUG_POISON_PTR(metadata->data);
  }
}

char *SmallCache::StartRead(blockid_t blockid) {
  Lock();

  Metadata *metadata = GetBlock_(blockid);
  metadata->lock_count++;

  Unlock();

  return metadata->data;
}

char *SmallCache::StartWrite(blockid_t blockid) {
  Lock();

  Metadata *metadata = GetBlock_(blockid);
  metadata->lock_count++;
  metadata->is_dirty = 1;

  Unlock();

  return metadata->data;
}

void SmallCache::StopRead(blockid_t blockid) {
  Lock();
  Metadata *metadata = GetBlock_(blockid);
  --metadata->lock_count;
  Unlock();
}

void SmallCache::StopWrite(blockid_t blockid) {
  StopRead(blockid);
}

void SmallCache::PerformCacheMiss_(blockid_t blockid) {
  char *data;
  Metadata *metadata = &metadata_[blockid];

  data = mem::Alloc<char>(n_block_bytes());
  inner_->Read(blockid, data);
  handler_->BlockThaw(data);
  metadata->data = data;
}

template<typename T>
class CacheArrayBlockActionHandler : public BlockActionHandler {
  FORBID_COPY(CacheArrayBlockActionHandler);

 private:
  char *default_elem_;
  index_t n_block_elems_;
  size_t n_elem_bytes_;

 public:
  CacheArrayBlockActionHandler() {
    n_block_elems_ = BIG_BAD_NUMBER;
    DEBUG_POISON_PTR(default_elem_);
  }

  ~CacheArrayBlockActionHandler() {
    mem::Free(default_elem_);
    n_block_elems_ = BIG_BAD_NUMBER;
    DEBUG_POISON_PTR(default_elem_);
  }

  void Init(index_t n_block_elems_in, const T& default_obj) {
    n_block_elems_ = n_block_elems_in;
    n_elem_bytes_ = ot::PointerFrozenSize(default_obj);
    default_elem_ = mem::Alloc<char>(n_elem_bytes_);
    ot::PointerFreeze(default_obj, default_elem_);
  }

  void BlockInit(char *block) {
    for (index_t i = 0; i < n_block_elems_; i++) {
      mem::CopyBytes(block, default_elem_, n_elem_bytes_);
      block += n_elem_bytes_;
    }
  }

  void BlockRefreeze(char *block) {
    for (index_t i = 0; i < n_block_elems_; i++) {
      ot::PointerRefreeze(reinterpret_cast<T*>(block));
      block += n_elem_bytes_;
    }
  }

  void BlockThaw(char *block) {
    for (index_t i = 0; i < n_block_elems_; i++) {
      ot::PointerThaw<T>(block);
      block += n_elem_bytes_;
    }
  }
};

template<typename T>
class CacheArray {
  FORBID_COPY(CacheArray);

 public:
  typedef T Element;

  enum Mode {
    /**
     * Only reading can be performed.
     */
    MODE_READ,
    /**
     * Writing can be performed, but it is not important if the data is
     * written back.
     */
    MODE_TEMP,
    /**
     * Writing can be performed, and the data should be written back.
     */
    MODE_WRITE
  };

 private:
  struct Metadata {
    Metadata() : data(NULL) {}
    char *data;
#ifdef DEBUG
    // lock count will be useful when we start using a FIFO layer
    char lock_count;
#endif
  };

 private:
  SmallCache *cache_;
  Mode mode_;
  index_t begin_;
  index_t end_;
  index_t begin_block_;
  index_t end_block_;
  unsigned int n_block_elems_;
  unsigned int n_elem_bytes_;
  ArrayList<Metadata> metadata_;

 public:
  CacheArray() {}
  ~CacheArray() {}

  void Init(SmallCache *cache_in, Mode mode_in,
      index_t begin_index_in, index_t end_index_in,
      index_t n_block_elems_in, size_t n_elem_bytes_in) {
    cache_ = cache_in;
    mode_ = mode_in;
    begin_ = begin_index_in;
    end_ = end_index_in;
    n_block_elems_ = n_block_elems_in;
    n_elem_bytes_ = n_elem_bytes_in;
    
    begin_block_ = begin_ / n_block_elems_;
    end_block_ = ((end_ + n_block_elems_ - 1) / n_block_elems_) + 1;
    
    metadata_.Init(end_block_ - begin_block_);

    DEBUG_ASSERT(mode_ == MODE_READ
        || mode_ == MODE_TEMP || mode_ == MODE_WRITE);
  }

  index_t begin_index() const {
    return begin_;
  }
  index_t end_index() const {
    return end_;
  }

  const Element *StartRead(index_t element_id) {
    DEBUG_ONLY(BoundsCheck_(element_id));
    return CheckoutElement_(element_id);
  }

  void StopRead(const Element *ptr, index_t element_id) {
    DEBUG_ONLY(BoundsCheck_(element_id));
    return CheckoutElement_(element_id);
  }

  Element *StartWrite(index_t element_id) {
    DEBUG_ONLY(BoundsCheck_(element_id));
    DEBUG_ASSERT(mode_ >= MODE_TEMP);
    ReleaseElement_(element_id);
  }

  void StopWrite(Element *ptr, index_t element_id) {
    DEBUG_ONLY(BoundsCheck_(element_id));
    DEBUG_ASSERT(mode_ >= MODE_TEMP);
    ReleaseElement_(element_id);
  }

  /**
   * Flushes all changes.
   */
  void Flush();

 private:
  void BoundsCheck_(index_t element_id) {
    DEBUG_ASSERT(element_id >= begin_);
    DEBUG_ASSERT(element_id < end_);
  }

  COMPILER_NOINLINE
  Element *HandleCacheMiss_(index_t element_id);

  Element *CheckoutElement_(index_t element_id) {
    DEBUG_ONLY(BoundsCheck_(element_id));

    BlockDevice::blockid_t blockid = element_id / n_block_elems_;
    BlockDevice::offset_t offset =
        uint(element_id % n_block_elems_) * n_elem_bytes_;
  
    Metadata *metadata = &metadata_[blockid - begin_block_];
    DEBUG_ONLY(metadata_->lock_count++);
    Element *ptr = reinterpret_cast<Element*>(metadata_->data + offset);

    if (unlikely(!metadata_->data)) {
      return HandleCacheMiss_(element_id);
    } else {
      return ptr;
    }
  }

  void ReleaseElement_(index_t element_id) {
    DEBUG_ONLY(BoundsCheck_(element_id));
    DEBUG_ONLY(--metadata_[element_id / n_block_elems_ - begin_block_].lock_count);
  }
};

template<typename T>
void CacheArray<T>::Flush() {
  BlockDev::blockid_t block_count = end_block_ - begin_block_;
  
  for (BlockDev::blockid_t i = 0; i < block_count; i++) {
    Metadata *metadata = &metadata_[i];
    
    DEBUG_ASSERT(metadata->lock_count == 0);
    
    if (unlikely(metadata->data != NULL)) {
      index_t blockid = i + begin_block_;
      
      if (mode_ == MODE_WRITE) {
        cache_->StopWrite(blockid);
      } else {
        cache_->StopRead(blockid);
      }
      metadata->data = NULL;
    }
  }
}

template<typename T>
typename CacheArray<T>::Element* CacheArray<T>::HandleCacheMiss_(
    index_t element_id) {
  BlockDevice::blockid_t blockid = element_id / n_block_elems_;
  Metadata *metadata = &metadata_[blockid];
  
  if (mode_ == MODE_WRITE) {
    metadata->data = cache_->StartWrite(blockid);
  } else {
    metadata->data = cache_->StartRead(blockid);
  }

  BlockDevice::offset_t offset =
      uint(element_id % n_block_elems_) * n_elem_bytes_;
  
  return reinterpret_cast<Element*>(metadata->data + offset);
}

#endif

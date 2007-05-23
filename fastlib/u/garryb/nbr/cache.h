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
  static const BlockDevice::blockid_t HEADER_BLOCKID = 0;

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

  /**
   * Create a SmallCache.
   *
   * If mode is READ or MODIFY, the incoming handler's InitFromMetadata method
   * will be called.
   *
   * If the mode is WRITE or TEMP, the handler is assumed to be initialized,
   * and its WriteMetadata will be called.
   */
  void Init(BlockDevice *inner_in, BlockActionHandler *handler_in,
     mode_t mode_in);

  char *StartRead(blockid_t blockid);
  char *StartWrite(blockid_t blockid);
  void StopRead(blockid_t blockid);
  void StopWrite(blockid_t blockid);
  void Flush(blockid_t begin_block, offset_t begin_offset,
    blockid_t last_block, offset_t end_offset);

  virtual void Read(blockid_t blockid,
      offset_t begin, offset_t end, char *data);
  virtual void Write(blockid_t blockid,
      offset_t begin, offset_t end, const char *data);

  virtual blockid_t AllocBlock() {
    blockid_t blockid = BlockDeviceWrapper::AllocBlock();
    metadatas_.Resize(n_blocks());
    return blockid;
  }
  
  BlockActionHandler *block_action_handler() const {
    return handler_;
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

/**
 * Array elements may vary in size from run to run.  However, we place the
 * constraint that each array element must be the same size, derived all
 * from the same default element.
 */
template<typename T>
class CacheArrayBlockActionHandler : public BlockActionHandler {
  FORBID_COPY(CacheArrayBlockActionHandler);

 private:
  ArrayList<char> default_elem_;
  
 public:
  CacheArrayBlockActionHandler() {}
  ~CacheArrayBlockActionHandler() {}

  void InitFromHeader(size_t header_size, char *header) {
    ArrayList<char> *default_elem_saved =
        ot::PointerThaw< ArrayList<char> >(header);
    default_elem_.Copy(*default_elem_saved);
  }

  void WriteHeader(size_t header_size, char *header) {
    DEBUG_ASSERT_MSG(header_size >= ot::PointerFrozenSize(default_elem_),
        "Block size too small -- "
        "At least 2 array elements should fit in a block.");
    // This looks funny, we're saving an ArrayList in an ArrayList.
    // However, it's important that we do so -- so we can store the
    // *size*!
    ot::PointerFreeze(default_elem_, header);
  }
  
  void Init(const T& default_obj) {
    default_elem_.Init(ot::PointerFrozenSize(default_obj));
    ot::PointerFreeze(default_obj, default_elem_.begin());
  }

  void BlockInitFrozen(size_t bytes, char *block) {
    index_t elems = bytes / default_elem_.size();
    for (index_t i = 0; i < elems; i++) {
      mem::CopyBytes(block, default_elem_.begin(), default_elem_.size());
      block += default_elem_.size();
    }
  }

  void BlockRefreeze(size_t bytes, const char *old_location, char *block) {
    index_t elems = bytes / default_elem_.size();
    for (index_t i = 0; i < elems; i++) {
      ot::PointerRefreeze(reinterpret_cast<const T*>(old_location), block);
      block += default_elem_.size();
      old_location += default_elem_.size();
    }
  }

  void BlockThaw(size_t bytes, char *block) {
    index_t elems = bytes / default_elem_.size();
    for (index_t i = 0; i < elems; i++) {
      ot::PointerThaw<T>(block);
      block += default_elem_.size();
    }
  }
  
  size_t n_elem_bytes() {
    return default_elem_.size();
  }
};

// LIMITATION: This type of cache array assumes that everything fits in
// memory (it never releases locks).
template<typename TElement>
class CacheArray {
  FORBID_COPY(CacheArray);

 public:
  typedef TElement Element;

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
  
 protected:
  /**
   * When dealing with the underlying cache, we must take into account
   * the metadata block.
   *
   * Within CacheArray all block ID's refer to "logical" not "physical"
   * block ID's, i.e. offset by one to account for metadata.
   */
  static const int BLOCK_OFFSET = 1;

 private:
  unsigned int n_block_elems_log_;
  unsigned int n_block_elems_mask_;
  ArrayList<Metadata> metadatas_;
  unsigned int n_elem_bytes_;
  
  index_t begin_;
  index_t end_;
  unsigned int n_block_elems_;
  
  /* Note these are fake block ID's, offset by 1, to account for metadata blocks. */
  BlockDevice::blockid_t begin_block_fake_;
  BlockDevice::blockid_t end_block_fake_;
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
    Init(other->cache_, mode_in, begin_index_in, end_index_in);
  }

  void Init(SmallCache *cache_in, BlockDevice::mode_t mode_in,
      index_t begin_index_in, index_t end_index_in);

  void Init(SmallCache *cache_in, BlockDevice::mode_t mode_in) {
    index_t block_elems =
        cache_in->n_block_bytes() /
        (static_cast<CacheArrayBlockActionHandler<T>*>
          (cache_in->block_action_handler()))->n_elem_bytes();
    Init(cache_in, mode_in, 0,
        cache_in->n_blocks() * block_elems - BLOCK_OFFSET);
  }

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
    if (block_computed != end_block_fake_) {
      end_block_fake_ = cache_->AllocBlock() - BLOCK_OFFSET;
      metadatas_.Resize(end_block_fake_ - begin_block_fake_);
      // Okay, notify the lower layers we're allocating.
      DEBUG_ASSERT_MSG(block_computed == end_block_fake_,
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
    BlockDevice::blockid_t blockid =
        (element_id >> n_block_elems_log_);

    Metadata *metadata = &metadatas_[blockid - begin_block_fake_];

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
        (element_id >> n_block_elems_log_) - begin_block_fake_].lock_count);
  }
};

template<typename T>
void CacheArray<T>::Init(SmallCache *cache_in, BlockDevice::mode_t mode_in,
    index_t begin_index_in, index_t end_index_in) {
  cache_ = cache_in;
  begin_ = begin_index_in;
  end_ = end_index_in;
  n_elem_bytes_ = (static_cast<CacheArrayBlockActionHandler<T>*>
      (cache_->block_action_handler()))->n_elem_bytes();
  DEBUG_ASSERT_MSG(cache_->n_block_bytes() % n_elem_bytes_ == 0,
      "Block size must be a multiple of element size.");
  n_block_elems_ = cache_->n_block_bytes() / n_elem_bytes_;
  // Cache size must be a power of 2.
  n_block_elems_log_ = math::IntLog2(n_block_elems_);
  n_block_elems_mask_ = n_block_elems_ - 1;
  
  begin_block_fake_ = begin_ / n_block_elems_;
  end_block_fake_ = (end_ + n_block_elems_ - 1) / n_block_elems_;
  
  metadatas_.Init(end_block_fake_ - begin_block_fake_);
  
  mode_ = mode_in;
}

template<typename T>
void CacheArray<T>::Flush() {
  for (BlockDevice::blockid_t blockid = begin_block_fake_;
      blockid < end_block_fake_; blockid++) {
    Metadata *metadata = &metadatas_[blockid - begin_block_fake_];
    if (metadata->data != NULL) {
      if (mode_ != BlockDevice::READ) {
        cache_->StopWrite(blockid + BLOCK_OFFSET);
      } else {
        cache_->StopRead(blockid + BLOCK_OFFSET);
      }
      DEBUG_SAME_INT(metadata->lock_count, 0);
      metadata->data = NULL;
    }
  }
  cache_->Flush(
      (begin_ >> n_block_elems_log_) + BLOCK_OFFSET,
      (begin_ & (n_block_elems_mask_)) * n_elem_bytes_,
      (end_ >> n_block_elems_log_) + BLOCK_OFFSET,
      (end_ & (n_block_elems_mask_)) * n_elem_bytes_);
}

template<typename T>
typename CacheArray<T>::Element* CacheArray<T>::HandleCacheMiss_(
    index_t element_id) {
  BlockDevice::blockid_t blockid = element_id >> n_block_elems_log_;
  Metadata *metadata = &metadatas_[blockid];
  
  if (mode_ != BlockDevice::READ) {
    metadata->data = cache_->StartWrite(blockid + BLOCK_OFFSET);
  } else {
    metadata->data = cache_->StartRead(blockid + BLOCK_OFFSET);
  }

  BlockDevice::offset_t offset =
      uint(element_id & (n_block_elems_ - 1)) * n_elem_bytes_;
  
  return reinterpret_cast<Element*>(metadata->data + offset);
}

/**
 * Specialed cache-array to simplify the creation/cleanup process.
 */
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

    null_device_.Init(
        (n_elems_in + n_block_elems_in + 1) / n_block_elems_in
          + CacheArray<T>::BLOCK_OFFSET,
        n_block_elems_in * handler->n_elem_bytes());
    underlying_cache_.Init(&null_device_, handler, BlockDevice::TEMP);
    
    CacheArray<T>::Init(&underlying_cache_, BlockDevice::TEMP, 0, n_elems_in);
  }
};

#endif

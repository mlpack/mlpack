#ifndef NBR_CACHEARRAY_H
#define NBR_CACHEARRAY_H

#include "cache.h"

/**
 * Array elements may vary in size from run to run.  However, we place the
 * constraint that each array element must be the same size, derived all
 * from the same default element.
 */
template<typename T>
class CacheArraySchema : public Schema {
  FORBID_COPY(CacheArraySchema);

 private:
  ArrayList<char> default_elem_;
  
 public:
  CacheArraySchema() {}
  ~CacheArraySchema() {}

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
  
  template<typename T> friend class CacheRead;
  template<typename T> friend class CacheWrite;
  template<typename T> friend class CacheReadIterator;

 public:
  typedef TElement Element;

 protected:
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

 protected:
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

 public:
  CacheArray() {}
  ~CacheArray() {}

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
        (static_cast<CacheArraySchema<TElement>*>
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
  
  SmallCache *cache() const {
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
    char *a = reinterpret_cast<char*>(StartWrite(index_a));
    char *b = reinterpret_cast<char*>(StartWrite(index_b));
    mem::Swap(a, b, n_elem_bytes_);
    ot::PointerRelocate<Element>(a, b);
    ot::PointerRelocate<Element>(b, a);
    ReleaseElement_(index_a);
    ReleaseElement_(index_b);
  }

  void Copy(index_t index_src, index_t index_dest) {
    DEBUG_ONLY(BoundsCheck_(index_src));
    DEBUG_ONLY(BoundsCheck_(index_dest));
    DEBUG_ASSERT(mode_ != BlockDevice::READ);
    const char *src = reinterpret_cast<char*>(StartWrite(index_src));
    char *dest = reinterpret_cast<char*>(StartWrite(index_dest));
    mem::Copy(dest, src, n_elem_bytes_);
    ot::PointerRelocate<Element>(src, dest);
    ReleaseElement_(index_src);
    ReleaseElement_(index_dest);
  }

  /**
   * Flushes all changes.
   */
  void Flush();

  /**
   * Flushes unclaimed parts of the first and last block to avoid fringe
   * cases.  To know why this is necessary, please contact Garry, or
   * experience the hard-to-track bug yourself.
   */
  void FixBoundaries();

  index_t Alloc() {
    BlockDevice::blockid_t block_computed = end_ >> n_block_elems_log_;
    end_++;
    if (block_computed >= end_block_fake_) {
      end_block_fake_ = block_computed + 1;
      metadatas_.Resize(end_block_fake_ - begin_block_fake_);
    }
    return end_ - 1;
  }

 private:
  void BoundsCheck_(index_t element_id) {
    DEBUG_BOUNDS(element_id - begin_, end_ - begin_);
  }

  BlockDevice::blockid_t Blockid_(index_t element_id) {
    return element_id >> n_block_elems_log_;
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
  
  void ReleaseBlock_(BlockDevice::blockid_t blockid) {
    DEBUG_ONLY(--metadatas_[blockid - begin_block_fake_].lock_count);
  }
  
  void ReleaseElement_(index_t element_id) {
    DEBUG_ONLY(BoundsCheck_(element_id));
    DEBUG_ONLY(
        ReleaseBlock_((element_id >> n_block_elems_log_)));
  }
};

template<typename TElement>
void CacheArray<TElement>::Init(SmallCache *cache_in, BlockDevice::mode_t mode_in,
    index_t begin_index_in, index_t end_index_in) {
  cache_ = cache_in;
  begin_ = begin_index_in;
  end_ = end_index_in;
  n_elem_bytes_ = (static_cast<CacheArraySchema<TElement>*>
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

template<typename TElement>
void CacheArray<TElement>::Flush() {
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
  if (mode_ != BlockDevice::READ) {
    // TODO: flushing isn't really done the way it needs to be done
    cache_->Flush(
        (begin_ >> n_block_elems_log_) + BLOCK_OFFSET,
        (begin_ & n_block_elems_mask_) * n_elem_bytes_,
        (end_ >> n_block_elems_log_) + BLOCK_OFFSET,
        (end_ & n_block_elems_mask_) * n_elem_bytes_);
  }
}

template<typename TElement>
void CacheArray<TElement>::FixBoundaries() {
  if (mode_ == BlockDevice::CREATE && begin_block_fake_ < end_block_fake_) {
    if ((begin_ & n_block_elems_mask_) != 0) {
      BlockDevice::blockid_t b = (begin_ >> n_block_elems_log_) + BLOCK_OFFSET;
      // Load the block full of the default element and flush the boundary.
      cache_->StartWrite(b);
      cache_->StopWrite(b);
      cache_->Flush(b, 0,
        b, (begin_ & n_block_elems_mask_) * n_elem_bytes_);
    }
    if ((end_ & n_block_elems_mask_) != 0) {
      BlockDevice::blockid_t b = (end_ >> n_block_elems_log_) + BLOCK_OFFSET;
      // Load the block full of the default element and flush the boundary.
      cache_->StartWrite(b);
      cache_->StopWrite(b);
      cache_->Flush(b, (end_ & n_block_elems_mask_) * n_elem_bytes_,
        b, n_block_elems_ * n_elem_bytes_);
    }
  }
}

template<typename TElement>
typename CacheArray<TElement>::Element* CacheArray<TElement>::HandleCacheMiss_(
    index_t element_id) {
  BlockDevice::blockid_t blockid = element_id >> n_block_elems_log_;
  Metadata *metadata = &metadatas_[blockid - begin_block_fake_];
  
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
template<typename TElement>
class TempCacheArray : public CacheArray<TElement> {
 private:
  SmallCache underlying_cache_;
  NullBlockDevice null_device_;
  
 public:
  ~TempCacheArray() {
    CacheArray<TElement>::Flush();
  }

  /** Creates a blank, temporary cached array */
  void Init(const TElement& default_obj,
      index_t n_elems_in,
      unsigned int n_block_elems_in) {
    CacheArraySchema<TElement> *handler =
        new CacheArraySchema<TElement>;
    handler->Init(default_obj);

    null_device_.Init(
        (n_elems_in + n_block_elems_in + 1) / n_block_elems_in
          + CacheArray<TElement>::BLOCK_OFFSET,
        n_block_elems_in * handler->n_elem_bytes());
    underlying_cache_.Init(&null_device_, handler, BlockDevice::TEMP);
    
    CacheArray<TElement>::Init(&underlying_cache_, BlockDevice::TEMP, 0, n_elems_in);
  }
};

template<typename Element>
class CacheRead {
  FORBID_COPY(CacheRead);

 private:
  const Element *element_;
#ifdef DEBUG
  CacheArray<Element> *cache_;
  BlockDevice::blockid_t blockid_;
#endif

 public:
  CacheRead(CacheArray<Element>* cache_in, index_t id) {
    element_ = cache_in->StartRead(id);
    DEBUG_ONLY(cache_ = cache_in);
    DEBUG_ONLY(blockid_ = cache_->Blockid_(id));
  }
  ~CacheRead() {
    DEBUG_ONLY(cache_->ReleaseBlock_(blockid_));
  }

  operator const Element * () const {
    return element_;
  }
  const Element * operator -> () const {
    return element_;
  }
  const Element & operator * () const {
    return *element_;
  }
};

template<typename Element>
class CacheWrite {
  FORBID_COPY(CacheWrite);

 private:
  Element *element_;
#ifdef DEBUG
  CacheArray<Element> *cache_;
  BlockDevice::blockid_t blockid_;
#endif

 public:
  CacheWrite(CacheArray<Element>* cache_in, index_t id) {
    element_ = cache_in->StartWrite(id);
    DEBUG_ONLY(cache_ = cache_in);
    DEBUG_ONLY(blockid_ = cache_->Blockid_(id));
  }
  ~CacheWrite() {
    DEBUG_ONLY(cache_->ReleaseBlock_(blockid_));
  }

  operator const Element * () const {
    return element_;
  }
  const Element * operator -> () const {
    return element_;
  }
  const Element & operator * () const {
    return *element_;
  }
  operator Element * () {
    return element_;
  }
  Element * operator -> () {
    return element_;
  }
  Element & operator * () {
    return *element_;
  }
};

template<typename Element>
class CacheReadIterator {
  FORBID_COPY(CacheReadIterator);

 private:
  const Element *element_;
  uint stride_;
  uint left_;
  CacheArray<Element> *cache_;
  BlockDevice::blockid_t blockid_;

 public:
  CacheReadIterator(CacheArray<Element>* cache_in, index_t begin_index) {
    cache_ = cache_in;
    blockid_ = begin_index >> cache_->n_block_elems_log_;
    element_ = cache_->StartRead(begin_index);
    stride_ = cache_->n_elem_bytes();
    unsigned int mask = cache_->n_block_elems_mask_;
    // equivalent to: block_size - (begin_index % block_size) - 1
    left_ = (begin_index & mask) ^ mask;
  }
  ~CacheReadIterator() {
    DEBUG_ONLY(cache_->ReleaseBlock_(blockid_));
  }

  operator const Element * () const {
    return element_;
  }
  const Element * operator -> () const {
    return element_;
  }
  const Element & operator * () const {
    return *element_;
  }
  
  void SetIndex(index_t begin_index) {
    DEBUG_ONLY(cache_->ReleaseBlock_(blockid_));
    blockid_ = begin_index >> cache_->n_block_elems_log_;
    element_ = cache_->StartRead(begin_index);
    unsigned int mask = cache_->n_block_elems_mask_;
    left_ = (begin_index & mask) ^ mask;
  }

  void Next() {
    element_ = mem::PointerAdd(element_, stride_);
    DEBUG_BOUNDS(left_, cache_->n_block_elems() + 1);
    if (unlikely(left_ == 0)) {
      left_ = cache_->n_block_elems();
      DEBUG_ONLY(cache_->ReleaseBlock_(blockid_));
      ++blockid_;
      element_ = cache_->StartRead(
          index_t(blockid_) << cache_->n_block_elems_log_);
    }
    --left_;
  }
};

#endif

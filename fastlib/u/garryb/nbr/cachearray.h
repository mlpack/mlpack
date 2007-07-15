#ifndef NBR_CACHEARRAY_H
#define NBR_CACHEARRAY_H

#include "cache.h"

/**
 * Array elements may vary in size from run to run.  However, we place the
 * constraint that each array element must be the same size, derived all
 * from the same default element.
 */
template<typename T>
class CacheArrayBlockHandler : public BlockHandler {
  FORBID_COPY(CacheArrayBlockHandler);

 private:
  enum { HEADER_BLOCKID = 0 };

 private:
  ArrayList<char> default_elem_;

 public:
  CacheArrayBlockHandler() {}
  ~CacheArrayBlockHandler() {}

  /**
   * Initializes this and sets up the block device's header.
   *
   * Do this before setting up the actual SmallCache or LRU cache, because
   * this allocates a block, circumventing the cache.
   */
  void Init(const T& default_obj) {
    default_elem_.Init(ot::PointerFrozenSize(default_obj));
    ot::PointerFreeze(default_obj, default_elem_.begin());
  }
  
  void WriteHeader(BlockDevice *inner_device) {
    // Next, we store the ArrayList in another ArrayList because we can't
    // get away with storing just the object (we would lose the size).
    ArrayList<char> buffer;
    buffer.Init(inner_device->n_block_bytes());
    size_t array_size = ot::PointerFrozenSize(default_elem_);
    (void) array_size;
    DEBUG_ASSERT_MSG(array_size <= inner_device->n_block_bytes(),
        "Too small of a block size, must be at least %ld bytes (obj is %ld)",
        long(array_size), long(default_elem_.size()));
    ot::PointerFreeze(default_elem_, buffer.begin());

    BlockDevice::blockid_t blockid = inner_device->AllocBlocks(1);
    (void) blockid;
    DEBUG_ASSERT_MSG(blockid == HEADER_BLOCKID, "Header block already exists");
    inner_device->Write(HEADER_BLOCKID, 0,
        inner_device->n_block_bytes(), buffer.begin());
  }

  /**
   * Inits from a block device -- using this on the cache itself will
   * probably cause lots of trouble (especially in non-read modes) so please
   * use it on the underlying block device.
   */
  void InitFromDevice(BlockDevice *inner_device) {
    ArrayList<char> buffer;

    buffer.Init(inner_device->n_block_bytes());
    // Read the first block, the header
    inner_device->Read(HEADER_BLOCKID, 0,
        inner_device->n_block_bytes(), buffer.begin());
    ArrayList<char> *default_elem_stored =
        ot::PointerThaw< ArrayList<char> >(buffer.begin());
    default_elem_.Copy(*default_elem_stored);
  }

  void BlockInitFrozen(BlockDevice::blockid_t blockid,
      BlockDevice::offset_t begin, BlockDevice::offset_t bytes, char *block) {
    if (blockid != HEADER_BLOCKID) {
      DEBUG_ASSERT((begin % default_elem_.size()) == 0);
      index_t elems = bytes / default_elem_.size();
      for (index_t i = 0; i < elems; i++) {
        mem::CopyBytes(block, default_elem_.begin(), default_elem_.size());
        block += default_elem_.size();
      }
    }
  }

  void BlockFreeze(BlockDevice::blockid_t blockid,
      BlockDevice::offset_t begin, BlockDevice::offset_t bytes,
      const char *old_location, char *block) {
    if (blockid != HEADER_BLOCKID) {
      DEBUG_ASSERT((begin % default_elem_.size()) == 0);
      index_t elems = bytes / default_elem_.size();
      for (index_t i = 0; i < elems; i++) {
        ot::PointerRefreeze(reinterpret_cast<const T*>(old_location), block);
        block += default_elem_.size();
        old_location += default_elem_.size();
      }
    }
  }

  void BlockThaw(BlockDevice::blockid_t blockid,
      BlockDevice::offset_t begin, BlockDevice::offset_t bytes,
      char *block) {
    if (blockid != HEADER_BLOCKID) {
      DEBUG_ASSERT(begin % default_elem_.size() == 0);
      index_t elems = bytes / default_elem_.size();
      for (index_t i = 0; i < elems; i++) {
        ot::PointerThaw<T>(block);
        block += default_elem_.size();
      }
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
  template<typename T, typename Q, typename R> friend class CacheIterImpl_;

 public:
  typedef TElement Element;

 protected:
  struct Metadata {
    Metadata()
     : data(NULL) {
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
  static const BlockDevice::blockid_t HEADER_BLOCKS = 1;

 protected:
  unsigned int n_block_elems_log_;
  unsigned int n_block_elems_mask_;
  ArrayList<Metadata> metadatas_;
  unsigned int n_elem_bytes_;
  BlockDevice::blockid_t skip_blocks_;

  index_t begin_;
  index_t next_alloc_;
  index_t end_;
  unsigned int n_block_elems_;

  BlockDevice::mode_t mode_;

  SmallCache *cache_;

 public:
  CacheArray() {}
  ~CacheArray() {}

  /** Reopens another cache array, the same range */
  void Init(CacheArray *other, BlockDevice::mode_t mode_in) {
    Init(other, mode_in, other->begin_index(), other->end_index());
  }

  /** Reopens another cache array, a sub-range only */
  void Init(CacheArray *other, BlockDevice::mode_t mode_in,
      index_t begin_index_in, index_t end_index_in) {
    Init(other->cache_, mode_in, begin_index_in, end_index_in);
  }

  /** Opens an existing SmallCache, a sub-range only (static use-case). */
  void Init(SmallCache *cache_in, BlockDevice::mode_t mode_in,
      index_t begin_index_in, index_t end_index_in);

  /**
   * Opens an existing SmallCache, a sub-range only.
   *
   * Behavior is inferred via the mode.
   */
  void Init(SmallCache *cache_in, BlockDevice::mode_t mode_in) {
    Init(cache_in, mode_in, 0, 0);
    Grow();
  }

  /**
   * Grows to at least the specified size.
   */
  void Grow(index_t end_element) {
    DEBUG_ASSERT_MSG(end_element >= end_,
        "end_element [%"LI"d] >= end_ [%"LI"d]",
        end_element, end_);
    end_ = end_element;
    next_alloc_ = end_element;
    metadatas_.Resize(((end_ + n_block_elems_ - 1) >> n_block_elems_log_)
        - skip_blocks_);
  }

  void Grow() {
    Grow((cache_->n_blocks() - HEADER_BLOCKS) << n_block_elems_log_);
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
    DEBUG_ASSERT(BlockDevice::can_write(mode_));
    return CheckoutElement_(element_id);
  }

  void StopRead(index_t element_id) {
    DEBUG_ONLY(BoundsCheck_(element_id));
    ReleaseElement_(element_id);
  }

  void StopWrite(index_t element_id) {
    DEBUG_ONLY(BoundsCheck_(element_id));
    DEBUG_ASSERT(BlockDevice::can_write(mode_));
    ReleaseElement_(element_id);
  }

  void Swap(index_t index_a, index_t index_b) {
    DEBUG_ONLY(BoundsCheck_(index_a));
    DEBUG_ONLY(BoundsCheck_(index_b));
    DEBUG_ASSERT(BlockDevice::can_write(mode_));
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
    DEBUG_ASSERT(BlockDevice::can_write(mode_));
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
  void Flush(bool clear);

  /**
   * Change the mdoe.
   */
  void Remode(BlockDevice::mode_t new_mode) {
    // TODO: Assert no blocks are open.
    mode_ = new_mode;
  }

  index_t Alloc(index_t count) {
    DEBUG_ASSERT(BlockDevice::is_dynamic(mode_));

    if (unlikely(next_alloc_ + count > end_)) {
      BlockDevice::blockid_t blocks_to_alloc =
          (count + n_block_elems_mask_) >> n_block_elems_log_;
      BlockDevice::blockid_t blockid = cache_->AllocBlocks(blocks_to_alloc);

      metadatas_.Resize(blockid + blocks_to_alloc
          - skip_blocks_ - HEADER_BLOCKS);

      next_alloc_ = (blockid - HEADER_BLOCKS) << n_block_elems_log_;
      end_ = next_alloc_ + (blocks_to_alloc << n_block_elems_log_);

      // If we straddle a block boundary, force the last block to be
      // dirty, so we avoid edge cases where part of the block is
      // initialized and the other isn't, and a crash occurs within
      // the block-handler when pulling in a block.
      if ((next_alloc_ & n_block_elems_mask_) != 0) {
        HandleCacheMiss_(end_ - 1);
      }
    }

    index_t ret_pos = next_alloc_;
    next_alloc_ += count;

    return ret_pos;
  }

  index_t Alloc() {
    DEBUG_ASSERT(BlockDevice::is_dynamic(mode_));

    if (unlikely(next_alloc_ >= end_)) {
      BlockDevice::blockid_t blockid = cache_->AllocBlocks(1);

      metadatas_.Resize(blockid - skip_blocks_ + (1 - HEADER_BLOCKS));

      next_alloc_ = (blockid - HEADER_BLOCKS) << n_block_elems_log_;
      end_ = next_alloc_ + n_block_elems_;

      // Force this block to be dirty.
      HandleCacheMiss_(next_alloc_);
    }

    index_t ret_pos = next_alloc_;
    next_alloc_++;

    return ret_pos;
  }

 private:
  void BoundsCheck_(index_t element_id) {
    DEBUG_BOUNDS(element_id - begin_, end_ - begin_);
  }

  BlockDevice::blockid_t Blockid_(index_t element_id) {
    return (element_id >> n_block_elems_log_) - skip_blocks_;
  }

  index_t FirstBlockElement_(BlockDevice::blockid_t fakeid) {
    return (fakeid + skip_blocks_) << n_block_elems_log_;
  }

  COMPILER_NOINLINE
  Element *HandleCacheMiss_(index_t element_id);

  Element *CheckoutElement_(index_t element_id) {
    DEBUG_ONLY(BoundsCheck_(element_id));

    BlockDevice::offset_t offset =
        (element_id & n_block_elems_mask_) * n_elem_bytes_;
    BlockDevice::blockid_t fakeid =
        Blockid_(element_id);

    Metadata *metadata = &metadatas_[fakeid];

    DEBUG_ONLY(metadata->lock_count++);

    if (unlikely(!metadata->data)) {
      return HandleCacheMiss_(element_id);
    } else {
      Element *ptr = reinterpret_cast<Element*>(metadata->data + offset);
      return ptr;
    }
  }

  void ReleaseBlock_(BlockDevice::blockid_t fakeid) {
    DEBUG_ONLY(--metadatas_[fakeid].lock_count);
  }

  void ReleaseElement_(index_t element_id) {
    DEBUG_ONLY(BoundsCheck_(element_id));
    DEBUG_ONLY(ReleaseBlock_(Blockid_(element_id)));
  }
};

template<typename TElement>
void CacheArray<TElement>::Init(
    SmallCache *cache_in, BlockDevice::mode_t mode_in,
    index_t begin_index_in, index_t end_index_in) {
  CacheArrayBlockHandler<TElement>* handler =
      static_cast<CacheArrayBlockHandler<TElement>*>(
          cache_in->block_handler());

  cache_ = cache_in;
  begin_ = begin_index_in;
  end_ = end_index_in;
  next_alloc_ = end_;
  mode_ = mode_in;

  n_elem_bytes_ = handler->n_elem_bytes();
  n_block_elems_ = cache_->n_block_bytes() / n_elem_bytes_;
  // Cache size must be a power of 2.
  n_block_elems_log_ = math::IntLog2(n_block_elems_);
  n_block_elems_mask_ = n_block_elems_ - 1;
  skip_blocks_ = begin_ / n_block_elems_;
  DEBUG_ASSERT_MSG(cache_->n_block_bytes() % n_elem_bytes_ == 0,
      "Block size must be a multiple of element size.");

  if (BlockDevice::need_init(mode_)) {
    handler->WriteHeader(cache_->inner());
    (void) cache_->AllocBlocks(0);
  }

  metadatas_.Init(((end_ + n_block_elems_ - 1) >> n_block_elems_log_)
      - skip_blocks_);
}

template<typename TElement>
void CacheArray<TElement>::Flush(bool clear) {
  for (index_t fakeid = 0;
      fakeid < metadatas_.size(); fakeid++) {
    Metadata *metadata = &metadatas_[fakeid];
    BlockDevice::blockid_t blockid = fakeid + HEADER_BLOCKS + skip_blocks_;

    if (metadata->data != NULL) {
      if (BlockDevice::can_write(mode_)) {
        cache_->StopWrite(blockid);
      } else {
        cache_->StopRead(blockid);
      }
      DEBUG_SAME_INT(metadata->lock_count, 0);
      metadata->data = NULL;
    }
  }
  if (BlockDevice::need_write(mode_)) {
    // TODO: flushing isn't really done the way it needs to be done
    cache_->Flush(
        clear,
        (begin_ >> n_block_elems_log_) + HEADER_BLOCKS,
        (begin_ & n_block_elems_mask_) * n_elem_bytes_,
        (end_ >> n_block_elems_log_) + HEADER_BLOCKS,
        (end_ & n_block_elems_mask_) * n_elem_bytes_);
  }
}

template<typename TElement>
typename CacheArray<TElement>::Element* CacheArray<TElement>::HandleCacheMiss_(
    index_t element_id) {
  BlockDevice::blockid_t fakeid = Blockid_(element_id);
  BlockDevice::blockid_t blockid = fakeid + HEADER_BLOCKS + skip_blocks_;
  Metadata *metadata = &metadatas_[fakeid];

  if (BlockDevice::can_write(mode_)) {
    metadata->data = cache_->StartWrite(blockid);
  } else {
    metadata->data = cache_->StartRead(blockid);
  }

  BlockDevice::offset_t offset =
      uint(element_id & (n_block_elems_mask_)) * n_elem_bytes_;

  return reinterpret_cast<Element*>(metadata->data + offset);
}

//------------------------------------------------------------------------

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

//------------------------------------------------------------------------

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

//------------------------------------------------------------------------

template<typename Helperclass, typename Element, typename BaseElement>
class CacheIterImpl_ {
  FORBID_COPY(CacheIterImpl_);

 private:
  Element *element_;
  uint stride_;
  uint left_;
  CacheArray<BaseElement> *cache_;
  BlockDevice::blockid_t blockid_;

 public:
  CacheIterImpl_(CacheArray<BaseElement>* cache_in, index_t begin_index) {
    cache_ = cache_in;
    blockid_ = cache_->Blockid_(begin_index);
    element_ = Helperclass::MyStartAccess_(cache_, begin_index);
    stride_ = cache_->n_elem_bytes();
    unsigned int mask = cache_->n_block_elems_mask_;
    // equivalent to: block_size - (begin_index % block_size) - 1
    left_ = (begin_index & mask) ^ mask;
  }
  ~CacheIterImpl_() {
    if (likely(element_ != NULL)) {
      cache_->ReleaseBlock_(blockid_);
    }
  }

  operator Element * () const {
    return element_;
  }
  Element * operator -> () const {
    return element_;
  }
  Element & operator * () const {
    return *element_;
  }

  void SetIndex(index_t begin_index) {
    DEBUG_ONLY(cache_->ReleaseBlock_(blockid_));
    blockid_ = cache_->Blockid_(begin_index);
    element_ = Helperclass::MyStartAccess_(cache_, begin_index);
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

      index_t elem_id = cache_->FirstBlockElement_(blockid_);
      element_ = Helperclass::MyStartAccess_(cache_, elem_id);
    }
    --left_;
  }
};

template<typename Element>
class CacheReadIterHelperclass_ {
 public:
  static const Element *MyStartAccess_(CacheArray<Element>* a, index_t i) {
    if (i < a->end_index()) {
      return a->StartRead(i);
    } else {
      return NULL;
    }
  }
};

template<typename Element>
class CacheReadIter
  : public CacheIterImpl_<CacheReadIterHelperclass_<Element>, const Element, Element> {
 public:
  CacheReadIter(CacheArray<Element>* cache_in, index_t begin_index)
      : CacheIterImpl_<CacheReadIterHelperclass_<Element>, const Element, Element>(
          cache_in, begin_index) {}
};

template<typename Element>
class CacheWriteIterHelperclass_ {
 public:
  static Element *MyStartAccess_(CacheArray<Element>* a, index_t i) {
    return a->StartWrite(i);
  }
};

template<typename Element>
class CacheWriteIter
  : public CacheIterImpl_<CacheWriteIterHelperclass_<Element>, Element, Element> {
 public:
  CacheWriteIter(CacheArray<Element>* cache_in, index_t begin_index)
      : CacheIterImpl_<CacheWriteIterHelperclass_<Element>, Element, Element>(
          cache_in, begin_index) {}
};

//------------------------------------------------------------------------

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
    CacheArray<TElement>::Flush(true);
  }

  /** Creates a blank, temporary cached array */
  void Init(const TElement& default_obj,
      index_t n_elems_in,
      unsigned int n_block_elems_in) {
    CacheArrayBlockHandler<TElement> *handler =
        new CacheArrayBlockHandler<TElement>;
    handler->Init(default_obj);

    null_device_.Init(0, n_block_elems_in * handler->n_elem_bytes());
    underlying_cache_.Init(&null_device_, handler, BlockDevice::M_TEMP);

    CacheArray<TElement>::Init(&underlying_cache_, BlockDevice::M_TEMP, 0, 0);

    if (n_elems_in != 0) {
      // Allocate a bunch of space.
      CacheArray<TElement>::Alloc(n_elems_in);
    }
  }
};


#endif

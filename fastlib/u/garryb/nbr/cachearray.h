#ifndef THOR_CACHEARRAY_H
#define THOR_CACHEARRAY_H

#include "distribcache.h"

/**
 * Array elements may vary in size from run to run.  However, we place the
 * constraint that each array element must be the same size, derived all
 * from the same default element.
 */
template<typename T>
class CacheArrayBlockHandler : public BlockHandler {
  FORBID_COPY(CacheArrayBlockHandler);

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
  
  void Serialize(ArrayList<char>* data) const {
    data->Copy(default_elem_);
  }
  
  void Deserialize(const ArrayList<char>& data) {
    default_elem_.Copy(data);
  }

  void BlockInitFrozen(BlockDevice::blockid_t blockid,
      BlockDevice::offset_t begin, BlockDevice::offset_t bytes, char *block) {
    DEBUG_ASSERT((begin % default_elem_.size()) == 0);
    index_t elems = bytes / default_elem_.size();
    for (index_t i = 0; i < elems; i++) {
      mem::CopyBytes(block, default_elem_.begin(), default_elem_.size());
      block += default_elem_.size();
    }
  }

  void BlockFreeze(BlockDevice::blockid_t blockid,
      BlockDevice::offset_t begin, BlockDevice::offset_t bytes,
      const char *old_location, char *block) {
    DEBUG_ASSERT(begin % default_elem_.size() == 0);
    index_t elems = bytes / default_elem_.size();
    for (index_t i = 0; i < elems; i++) {
      ot::PointerRefreeze(reinterpret_cast<const T*>(old_location), block);
      block += default_elem_.size();
      old_location += default_elem_.size();
    }
  }

  void BlockThaw(BlockDevice::blockid_t blockid,
      BlockDevice::offset_t begin, BlockDevice::offset_t bytes,
      char *block) {
    DEBUG_ASSERT(begin % default_elem_.size() == 0);
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

 protected:
  struct Metadata {
    Metadata() : data(NULL) {
      lock_count = 0;
    }
    char *data;
    int lock_count;
  };

 protected:
  /**
   * Number of pages in the thread-local FIFO cache.
   *
   * This absolutely cannot be less than the maximum number of concurrent
   * "locked" blocks!  This maximum should be 32 for most use cases, with 64
   * giving a nice balance between efficiency and memory usage -- given the
   * worst-case of 32 used blocks, the mean search time for an empty FIFO
   * entry is two, and only 64 blocks are forced into RAM.
   */
  static const int FIFO_SIZE = 64;
  /** Bitmask for doing modulo FIFO_SIZE. */
  static const int FIFO_MASK = (FIFO_SIZE-1);

 protected:
  Metadata *adjusted_metadatas_;
  unsigned int n_block_elems_log_;
  unsigned int n_block_elems_mask_;

  ArrayList<Metadata> metadatas_;

  BlockDevice::blockid_t *fifo_;
  int fifo_index_;

  unsigned int n_elem_bytes_;
  index_t begin_;
  index_t next_alloc_;
  index_t end_;

  BlockDevice::blockid_t skip_blocks_;
  BlockDevice::mode_t mode_;

  DistributedCache *cache_;

 public:
  /** Helper to help you create a DistributedCache. */
  static void InitDistributedCacheMaster(int channel,
      index_t n_block_elems, const Element& default_elem, size_t total_ram,
      DistributedCache *cache) {
    CacheArrayBlockHandler<Element> *handler =
        new CacheArrayBlockHandler<Element>();
    handler->Init(default_elem);
    cache->InitMaster(channel, n_block_elems * handler->n_elem_bytes(),
        total_ram, handler);
  }
  /** Helper to help you connect a DistributedCache to master. */
  static void InitDistributedCacheWorker(int channel,
      size_t total_ram, DistributedCache *cache) {
    CacheArrayBlockHandler<Element> *handler =
        new CacheArrayBlockHandler<Element>();
    cache->InitWorker(channel, total_ram, handler);
  }

 public:
  CacheArray() {}
  ~CacheArray() {
    Flush();
    mem::Free(fifo_);
  }

  /** Reopens another cache array, the same range */
  void Init(CacheArray *other, BlockDevice::mode_t mode_in) {
    Init(other, mode_in, other->begin_index(), other->end_index());
  }

  /** Reopens another cache array, a sub-range only */
  void Init(CacheArray *other, BlockDevice::mode_t mode_in,
      index_t begin_index_in, index_t end_index_in) {
    Init(other->cache_, mode_in, begin_index_in, end_index_in);
  }

  /** Opens an existing DistributedCache, a sub-range only (static use-case). */
  void Init(DistributedCache *cache_in, BlockDevice::mode_t mode_in,
      index_t begin_index_in, index_t end_index_in);

  /**
   * Opens an existing DistributedCache, a sub-range only.
   *
   * Behavior is inferred via the mode.
   */
  void Init(DistributedCache *cache_in, BlockDevice::mode_t mode_in) {
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
    metadatas_.Resize(((end_ + n_block_elems_mask()) >> n_block_elems_log())
        - skip_blocks_);
    adjusted_metadatas_ = metadatas_.begin() - skip_blocks_;
    MarkRanges_();
  }

  void Grow() {
    Grow(cache_->n_blocks() << n_block_elems_log());
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

  unsigned int n_block_elems_log() const {
    return n_block_elems_log_;
  }
  unsigned int n_block_elems() const {
    return 1 << n_block_elems_log_;
  }
  unsigned int n_block_elems_mask() const {
    return n_block_elems_mask_;
  }

  DistributedCache *cache() const {
    return cache_;
  }

  const Element *StartRead(index_t element_id) {
    return CheckoutElement_(element_id);
  }

  Element *StartWrite(index_t element_id) {
    DEBUG_ASSERT(BlockDevice::can_write(mode_));
    return CheckoutElement_(element_id);
  }

  void Flush();

  void StopRead(index_t element_id) {
    DEBUG_ONLY(BoundsCheck_(element_id));
    ReleaseElement(element_id);
  }

  void StopWrite(index_t element_id) {
    DEBUG_ONLY(BoundsCheck_(element_id));
    DEBUG_ASSERT(BlockDevice::can_write(mode_));
    ReleaseElement(element_id);
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
    ReleaseElement(index_a);
    ReleaseElement(index_b);
  }

  void Copy(index_t index_src, index_t index_dest) {
    DEBUG_ONLY(BoundsCheck_(index_src));
    DEBUG_ONLY(BoundsCheck_(index_dest));
    DEBUG_ASSERT(BlockDevice::can_write(mode_));
    const char *src = reinterpret_cast<char*>(StartWrite(index_src));
    char *dest = reinterpret_cast<char*>(StartWrite(index_dest));
    mem::Copy(dest, src, n_elem_bytes_);
    ot::PointerRelocate<Element>(src, dest);
    ReleaseElement(index_src);
    ReleaseElement(index_dest);
  }

  index_t AllocD(int owner, index_t count) {
    DEBUG_ASSERT(BlockDevice::is_dynamic(mode_));

    if (unlikely(next_alloc_ + count > end_)) {
      BlockDevice::blockid_t blocks_to_alloc =
          (count + n_block_elems_mask()) >> n_block_elems_log();
      BlockDevice::blockid_t blockid = cache_->AllocBlocks(
          blocks_to_alloc, owner);

      metadatas_.Resize(blockid + blocks_to_alloc - skip_blocks_);
      adjusted_metadatas_ = metadatas_.begin() - skip_blocks_;

      next_alloc_ = blockid << n_block_elems_log();
      end_ = next_alloc_ + (blocks_to_alloc << n_block_elems_log());
      MarkRanges_();
    }

    index_t ret_pos = next_alloc_;
    next_alloc_ += count;

    return ret_pos;
  }

  index_t AllocD(int owner) {
    DEBUG_ASSERT(BlockDevice::is_dynamic(mode_));

    if (unlikely(next_alloc_ >= end_)) {
      BlockDevice::blockid_t blockid = cache_->AllocBlocks(1, owner);

      metadatas_.Resize(blockid - skip_blocks_ + 1);
      adjusted_metadatas_ = metadatas_.begin() - skip_blocks_;

      next_alloc_ = blockid << n_block_elems_log();
      end_ = next_alloc_ + n_block_elems();
      MarkRanges_();
    }

    index_t ret_pos = next_alloc_;
    next_alloc_++;

    return ret_pos;
  }

 private:
  void BoundsCheck_(index_t element_id) {
    DEBUG_BOUNDS(element_id - begin_, end_ - begin_);
  }

  COMPILER_NOINLINE
  Element *HandleCacheMiss_(index_t element_id);

  // TODO: Think about how this affects register pressure
  Element *CheckoutElement_(index_t element_id) {
    DEBUG_ONLY(BoundsCheck_(element_id));

    Metadata *metadata = (element_id >> n_block_elems_log())
        + adjusted_metadatas_;
    char *data = metadata->data;
    BlockDevice::offset_t offset = Offset(element_id);


    ++metadata->lock_count;

    if (likely(data != NULL)) {
      return reinterpret_cast<Element*>(data + offset);
    } else {
      return HandleCacheMiss_(element_id);
    }
  }

  void MarkRanges_() {
    if (!BlockDevice::is_dynamic(mode_)) {
      cache_->AddPartialDirtyRange(
          Blockid(begin_), Offset(begin_),
          Blockid(end_), Offset(end_));
    }
  }

 public:
  /* these are public so various classes can use them efficiently */

  void ReleaseBlock(BlockDevice::blockid_t blockid) {
    --adjusted_metadatas_[blockid].lock_count;
  }

  index_t BlockElement(BlockDevice::blockid_t blockid) {
    return blockid << n_block_elems_log();
  }

  BlockDevice::blockid_t Blockid(index_t element_id) {
    return element_id >> n_block_elems_log();
  }
  
  BlockDevice::offset_t Offset(index_t element_id) {
    return (element_id & n_block_elems_mask()) * n_elem_bytes_;
  }

  void ReleaseElement(index_t element_id) {
    DEBUG_ONLY(BoundsCheck_(element_id));
    ReleaseBlock(Blockid(element_id));
  }
};

template<typename TElement>
void CacheArray<TElement>::Init(
    DistributedCache *cache_in, BlockDevice::mode_t mode_in,
    index_t begin_index_in, index_t end_index_in) {
  CacheArrayBlockHandler<TElement>* handler =
      static_cast<CacheArrayBlockHandler<TElement>*>(
          cache_in->block_handler());

  cache_ = cache_in;
  begin_ = begin_index_in;
  end_ = end_index_in;
  DEBUG_ASSERT(end_ >= begin_);
  DEBUG_ASSERT(begin_ >= 0);
  next_alloc_ = end_;
  mode_ = mode_in;
  n_elem_bytes_ = handler->n_elem_bytes();
  fifo_ = mem::Alloc<BlockDevice::blockid_t>(FIFO_SIZE);
  mem::ConstructAll(fifo_, -1, FIFO_SIZE);
  fifo_index_ = 0;

  unsigned n_block_elems_calc = cache_->n_block_bytes() / n_elem_bytes_;
  // Cache size must be a power of 2.
  n_block_elems_log_ = math::IntLog2(n_block_elems_calc);
  n_block_elems_mask_ = n_block_elems_calc - 1;
  skip_blocks_ = begin_ / n_block_elems_calc;
  DEBUG_ASSERT_MSG(cache_->n_block_bytes() % n_elem_bytes_ == 0,
      "Block size must be a multiple of element size.");

  MarkRanges_();

  metadatas_.Init(((end_ + n_block_elems_mask()) >> n_block_elems_log())
      - skip_blocks_);
  adjusted_metadatas_ = metadatas_.begin() - skip_blocks_;
}

template<typename TElement>
void CacheArray<TElement>::Flush() {
  for (int i = 0; i < FIFO_SIZE; i++) {
    BlockDevice::blockid_t blockid = fifo_[i];

    if (blockid >= 0) {
      Metadata *metadata = adjusted_metadatas_ + blockid;

      if (BlockDevice::can_write(mode_)) {
        cache_->StopWrite(blockid);
      } else {
        cache_->StopRead(blockid);
      }

      DEBUG_SAME_INT(metadata->lock_count, 0);
      metadata->data = NULL;
      fifo_[i] = -1;
    }
  }
}

template<typename TElement>
typename CacheArray<TElement>::Element* CacheArray<TElement>::HandleCacheMiss_(
    index_t element_id) {
  BlockDevice::blockid_t victim;
  Metadata *victim_metadata;

  // warning, this isn't very readable... basically, look for the first
  // unlocked item -- the most likely case is that the first item in the
  // fifo is non-negative (i.e. it exists) and it's most likely not locked
  for (;;) {
    fifo_index_ = (fifo_index_+1) & FIFO_MASK;
    victim = fifo_[fifo_index_];
    if (unlikely(victim < 0)) {
      break;
    }
    victim_metadata = adjusted_metadatas_ + victim;
    if (unlikely(victim_metadata->lock_count != 0)) {
      continue;
    }
    DEBUG_ASSERT(victim_metadata->data != NULL);
    if (BlockDevice::can_write(mode_)) {
      cache_->StopWrite(victim);
    } else {
      cache_->StopRead(victim);
    }
    victim_metadata->data = NULL;
    break;
  }

  BlockDevice::blockid_t blockid = Blockid(element_id);
  Metadata *metadata = adjusted_metadatas_ + blockid;
  
  fifo_[fifo_index_] = blockid;

  if (BlockDevice::can_write(mode_)) {
    metadata->data = cache_->StartWrite(blockid,
        !BlockDevice::is_dynamic(mode_));
    //putchar('#');
  } else {
    metadata->data = cache_->StartRead(blockid);
    //putchar('.');
  }

  BlockDevice::offset_t offset =
      uint(element_id & (n_block_elems_mask())) * n_elem_bytes_;

  return reinterpret_cast<Element*>(metadata->data + offset);
}

//------------------------------------------------------------------------

template<typename Element>
class CacheRead {
  FORBID_COPY(CacheRead);

 private:
  const Element *element_;
  CacheArray<Element> *cache_;
  BlockDevice::blockid_t blockid_;

 public:
  CacheRead(CacheArray<Element>* cache_in, index_t id) {
    element_ = cache_in->StartRead(id);
    cache_ = cache_in;
    blockid_ = cache_->Blockid(id);
  }
  ~CacheRead() {
    cache_->ReleaseBlock(blockid_);
  }

  const Element *get() const {
    return element_;
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
  CacheArray<Element> *cache_;
  BlockDevice::blockid_t blockid_;

 public:
  CacheWrite(CacheArray<Element>* cache_in, index_t id) {
    element_ = cache_in->StartWrite(id);
    cache_ = cache_in;
    blockid_ = cache_->Blockid(id);
  }
  ~CacheWrite() {
    cache_->ReleaseBlock(blockid_);
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
    blockid_ = cache_->Blockid(begin_index);
    element_ = Helperclass::MyStartAccess_(cache_, begin_index);
    stride_ = cache_->n_elem_bytes();
    unsigned int mask = cache_->n_block_elems_mask();
    // equivalent to: block_size - (begin_index % block_size) - 1
    left_ = (begin_index ^ mask) & mask;
  }
  ~CacheIterImpl_() {
    if (likely(element_ != NULL)) {
      cache_->ReleaseBlock(blockid_);
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
    cache_->ReleaseBlock(blockid_);
    blockid_ = cache_->Blockid(begin_index);
    element_ = Helperclass::MyStartAccess_(cache_, begin_index);
    unsigned int mask = cache_->n_block_elems_mask();
    left_ = (begin_index ^ mask) & mask;
  }

  void Next() {
    DEBUG_BOUNDS(left_, cache_->n_block_elems() + 1);
    element_ = mem::PointerAdd(element_, stride_);
    if (unlikely(left_ == 0)) {
      NextBlock_();
      return;
    }
    --left_;
  }

 private:
  COMPILER_NOINLINE
  void NextBlock_();
};

template<typename Helperclass, typename Element, typename BaseElement>
void CacheIterImpl_<Helperclass, Element, BaseElement>::NextBlock_() {
  left_ = cache_->n_block_elems_mask();
  cache_->ReleaseBlock(blockid_);
  ++blockid_;

  index_t elem_id = cache_->BlockElement(blockid_);
  if (likely(elem_id < cache_->end_index())) {
    element_ = Helperclass::MyStartAccess_(cache_, elem_id);
  } else {
    element_ = NULL;
  }
}

template<typename Element>
class CacheReadIterHelperclass_ {
 public:
  static const Element *MyStartAccess_(CacheArray<Element>* a, index_t i) {
    return a->StartRead(i);
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
 * Condensed-RAM array, .
 */
template<typename TElement>
class SubsetArray {
  FORBID_COPY(SubsetArray);

 public:
  typedef TElement Element;

 private:
  size_t n_elem_bytes_;
  char *adjusted_;
  index_t begin_;
  index_t end_;

 public:
  SubsetArray() {}
  ~SubsetArray() {
    mem::Free(&(*this)[begin_]);
  }

  void Init(const Element& default_elem, index_t begin, index_t end) {
    n_elem_bytes_ = ot::PointerFrozenSize(default_elem);
    begin_ = begin;
    end_ = end;
    adjusted_ = NULL;
    if (begin_ < end_) {
      char *base = mem::Alloc<char>(n_elem_bytes_ * (end - begin));
      char *adjusted = base - (begin * n_elem_bytes_);
      ot::PointerFreeze(default_elem, base);
      for (index_t i = begin + 1; i < end; i++) {
        char *ptr = adjusted + i * n_elem_bytes_;
        mem::CopyBytes(ptr, base, n_elem_bytes_);
        ot::PointerThaw<Element>(ptr);
      }
      ot::PointerThaw<Element>(base);
      adjusted_ = adjusted;
    }
  }

  index_t n_elem_bytes() const {
    return n_elem_bytes_;
  }
  
  const Element& operator[] (index_t i) const {
    return *reinterpret_cast<Element*>(adjusted_ + i * n_elem_bytes_);
  }

  Element& operator[] (index_t i) {
    return *reinterpret_cast<Element*>(adjusted_ + i * n_elem_bytes_);
  }
};

//#error what *is* a TempCache now?
///**
// * Specialed cache-array to simplify the creation/cleanup process.
// */
//template<typename TElement>
//class TempCacheArray : public CacheArray<TElement> {
// private:
//  DistributedCache underlying_cache_;
//  NullBlockDevice null_device_;
//
// public:
//  ~TempCacheArray() {
//    CacheArray<TElement>::Flush(true);
//  }
//
//  /** Creates a blank, temporary cached array */
//  void Init(const TElement& default_obj,
//      index_t n_elems_in,
//      unsigned int n_block_elems_in,
//      size_t total_ram = 16777216) {
//    CacheArrayBlockHandler<TElement> *handler =
//        new CacheArrayBlockHandler<TElement>;
//    handler->Init(default_obj);
//
//    null_device_.Init(0, n_block_elems_in * handler->n_elem_bytes());
//    underlying_cache_.InitMaster(&null_device_, handler, BlockDevice::M_TEMP);
//
//    CacheArray<TElement>::Init(&underlying_cache_, BlockDevice::M_TEMP, 0, 0);
//
//    if (n_elems_in != 0) {
//      // Allocate a bunch of space.
//      CacheArray<TElement>::Alloc(n_elems_in);
//    }
//  }
//};

#endif

//  void WriteHeader(BlockDevice *inner_device) {
//    // Next, we store the ArrayList in another ArrayList because we can't
//    // get away with storing just the object (we would lose the size).
//    ArrayList<char> buffer;
//    buffer.Init(inner_device->n_block_bytes());
//    size_t array_size = ot::PointerFrozenSize(default_elem_);
//    (void) array_size;
//    DEBUG_ASSERT_MSG(array_size <= inner_device->n_block_bytes(),
//        "Too small of a block size, must be at least %ld bytes (obj is %ld)",
//        long(array_size), long(default_elem_.size()));
//    ot::PointerFreeze(default_elem_, buffer.begin());
//
//    BlockDevice::blockid_t blockid = inner_device->AllocBlocks(1);
//    (void) blockid;
//    DEBUG_ASSERT_MSG(blockid == HEADER_BLOCKID, "Header block already exists");
//    inner_device->Write(HEADER_BLOCKID, 0,
//        inner_device->n_block_bytes(), buffer.begin());
//  }

//  /**
//   * Inits from a block device -- using this on the cache itself will
//   * probably cause lots of trouble (especially in non-read modes) so please
//   * use it on the underlying block device.
//   */
//  void InitFromDevice(BlockDevice *inner_device) {
//    ArrayList<char> buffer;
//
//    buffer.Init(inner_device->n_block_bytes());
//    // Read the first block, the header
//    inner_device->Read(HEADER_BLOCKID, 0,
//        inner_device->n_block_bytes(), buffer.begin());
//    ArrayList<char> *default_elem_stored =
//        ot::PointerThaw< ArrayList<char> >(buffer.begin());
//    default_elem_.Copy(*default_elem_stored);
//  }

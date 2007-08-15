/**
 * @file cachearray.h
 *
 * A cached array based on the THOR distributed cache.
 */

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
  void Init(const T& default_obj);
  
  void Serialize(ArrayList<char>* data) const;
  
  void Deserialize(const ArrayList<char>& data);

  void BlockInitFrozen(BlockDevice::blockid_t blockid,
      BlockDevice::offset_t begin, BlockDevice::offset_t bytes, char *block);

  void BlockFreeze(BlockDevice::blockid_t blockid,
      BlockDevice::offset_t begin, BlockDevice::offset_t bytes,
      const char *old_location, char *block);

  void BlockThaw(BlockDevice::blockid_t blockid,
      BlockDevice::offset_t begin, BlockDevice::offset_t bytes,
      char *block);

  /** Gets the number of bytes for an element. */
  size_t n_elem_bytes() {
    return default_elem_.size();
  }

  /** Gets the contents of the default initial element. */
  void GetDefaultElement(T *default_element_out);
};

/**
 * A cached array.
 *
 * The cached array metaphor is an array of objects (which may contain
 * points as long as they are serializable via object traversal) which are
 * stored in blocks and brought into RAM as needed.
 *
 * A cached array depends on an underlying cache, DistributedCache, which
 * manages page.  The cached array only locks into RAM a small subset,
 * controlled by FIFO_SIZE, of pages.
 *
 * A cached array is purposely not thread-safe, but it is fine to have many
 * cached arrays accessing the same distributed cache as long as they are
 * not read-write or write-write dependencies for the same region of elements.
 */
template<typename T>
class CacheArray {
  FORBID_COPY(CacheArray);

 public:
  /** The type of an element. */
  typedef T Element;

 protected:
  /** Per-block metadata. */
  struct Metadata {
    /** The in-core memory for the block. */
    char *data;
    /** The number of times this block is locked into this cached array. */
    int lock_count;

    /** Default constructor (called during array resizing). */
    Metadata() {
      data = NULL;
      lock_count = 0;
    }
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
  /** The metadata array, but adjusted (see how it is used in code). */
  Metadata *adjusted_metadatas_;
  /** The log base 2 of the number of elements per block. */
  unsigned int n_block_elems_log_;
  /** The bit mask of the number of elements per block. */
  unsigned int n_block_elems_mask_;

  /** The metadatas array. */
  ArrayList<Metadata> metadatas_;

  /** The circular fixed-size FIFO queue of blocks that are locked in memory. */
  BlockDevice::blockid_t *fifo_;
  /** The index in the circular FIFO. */
  int fifo_index_;

  /** Number of bytes per element. */
  unsigned int n_elem_bytes_;
  /** The first element. */
  index_t begin_;
  /** The next element to allocate. */
  index_t next_alloc_;
  /** The last element. */
  index_t end_;

  /**
   * Number of blocks that are skipped before the beginning of
   * the metadata array.
   */
  BlockDevice::blockid_t skip_blocks_;
  /** The mode in which this is being accessed. */
  BlockDevice::mode_t mode_;

  /** The underlying distributed cache. */
  DistributedCache *cache_;

 public:
  /**
   * Finds a power-of-two block size that is at most the specified number
   * of kilobytes.
   */
  static index_t ConvertBlockSize(const Element& element, int kilobytes);
  /** Helper to help you create a DistributedCache. */
  static void CreateCacheMaster(int channel,
      index_t n_block_elems, const Element& default_elem, double megs,
      DistributedCache *cache);

  /** Helper to help you connect a DistributedCache to master. */
  static void CreateCacheWorker(int channel, double megs,
      DistributedCache *cache);

  /**
   * Gets the default element of a cache.
   */
  static void GetDefaultElement(DistributedCache *cache, Element *element) {
    static_cast<CacheArrayBlockHandler<T>*>(
        cache->block_handler())->GetDefaultElement(element);
  }

  /**
   * Gets size of an element.
   */
  static size_t GetNumElementBytes(DistributedCache *cache) {
    return static_cast<CacheArrayBlockHandler<T>*>(
        cache->block_handler())->n_elem_bytes();
  }

  /**
   * Gets the number of elements in a block.
   */
  static size_t GetNumBlockElements(DistributedCache *cache) {
    return cache->n_block_bytes() / GetNumElementBytes(cache);
  }

 public:
  CacheArray() {}
  ~CacheArray() {
    Flush();
    mem::Free(fifo_);
  }

  void InitCreate(int channel, index_t n_block_elems,
      const Element& default_elem, double megs, DistributedCache *cache_in) {
    CreateCacheMaster(channel, n_block_elems, default_elem,
        megs, cache_in);
    Init(cache_in, BlockDevice::M_CREATE);
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
  void Grow(index_t end_element);

  /**
   * Grows to the size of the underlying cache.
   *
   * Warning: Some of the edge elements might still be uninitialized.
   */
  void Grow();

  /**
   * Gets the first element index this cache array is working on.
   */
  index_t begin_index() const {
    return begin_;
  }
  /**
   * Gets one past the last index this cache array is working on.
   */
  index_t end_index() const {
    return end_;
  }

  /**
   * Gets the size in bytes of an element.
   */
  unsigned int n_elem_bytes() const {
    return n_elem_bytes_;
  }

  /**
   * Gets the log base 2 of the number of elements in a block.
   */
  unsigned int n_block_elems_log() const {
    return n_block_elems_log_;
  }

  /**
   * Gets the number of elements in a block.
   */
  index_t n_block_elems() const {
    return n_block_elems_mask_ + 1;
  }

  /**
   * Gets the bit-mask that for figuring out the block offset of an element.
   */
  unsigned int n_block_elems_mask() const {
    return n_block_elems_mask_;
  }

  /**
   * Gets the underlying distributed cache.
   */
  DistributedCache *cache() const {
    return cache_;
  }

  /**
   * Checks out an element for reading.
   */
  const Element *StartRead(index_t element_id) {
    return CheckoutElement_(element_id);
  }

  /**
   * Checks out an element for writing.
   */
  Element *StartWrite(index_t element_id) {
    DEBUG_ASSERT(BlockDevice::can_write(mode_));
    return CheckoutElement_(element_id);
  }


  void StopRead(index_t element_id) {
    DEBUG_ONLY(BoundsCheck_(element_id));
    ReleaseElement(element_id);
  }

  void StopWrite(index_t element_id) {
    DEBUG_ONLY(BoundsCheck_(element_id));
    DEBUG_ASSERT(BlockDevice::can_write(mode_));
    ReleaseElement(element_id);
  }

  /**
   * Unlocks the blocks currently held in the FIFO so that the underlying
   * cache is free to flush them.
   */
  void Flush();

  /**
   * Swaps two elements (and what they point to).
   *
   * @param index_a the index of one element to swap
   * @param index_b the index of the other element to swap
   */
  void Swap(index_t index_a, index_t index_b);

  /**
   * Copies an element to another element, overwriting.
   *
   * @param index_src the index to copy from
   * @param index_dest the index to copy to, overwriting
   */
  void Copy(index_t index_src, index_t index_dest);

  /**
   * Allocates elements, with a preferred owner.
   *
   * If a new block must be allocated it will be assigned to the specified
   * owner.
   *
   * @param owner the owner to assign any new block to
   * @param count the number of elements to allocate
   */
  index_t AllocD(int owner, index_t count);

  /**
   * Allocates a single element, with a preferred owner.
   *
   * If a new block must be allocated it will be assigned to the specified
   * owner.
   *
   * @param owner the owner to assign any new block to
   */
  index_t AllocD(int owner);

  /** Releases a block explicitly, like StopRead or StopWrite. */
  void ReleaseBlock(BlockDevice::blockid_t blockid) {
    --adjusted_metadatas_[blockid].lock_count;
  }

  /** Gets the first element from a block ID. */
  index_t BlockElement(BlockDevice::blockid_t blockid) {
    return blockid << n_block_elems_log();
  }

  /** Gets the block ID from an element. */
  BlockDevice::blockid_t Blockid(index_t element_id) {
    return element_id >> n_block_elems_log();
  }

  /** Gets the within-block byte offset of an element. */
  BlockDevice::offset_t Offset(index_t element_id) {
    return (element_id & n_block_elems_mask()) * n_elem_bytes_;
  }

  /** Releases an element, same as StopRead or StopWrite. */
  void ReleaseElement(index_t element_id) {
    DEBUG_ONLY(BoundsCheck_(element_id));
    ReleaseBlock(Blockid(element_id));
  }

 private:
  /** Does a bounds check on an element ID. */
  void BoundsCheck_(index_t element_id) {
    DEBUG_BOUNDS(element_id - begin_, end_ - begin_);
  }

  /** Handles a miss from the internal FIFO. */
  COMPILER_NOINLINE
  Element *HandleCacheMiss_(index_t element_id);

  /** Checks out an element (happy-path). */
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

  /**
   * Marks our write ranges if we're in a mode that requires partially dirty
   * ranges (see documentation for DistributedCache).
   */
  void MarkRanges_();
};

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
 * Condensed-RAM array.
 *
 * This stores data in a flattened format, but there is no caching that
 * goes on because it is assumed this will fit in RAM.
 *
 * Only handles object serializable via object traversal (i.e. with OT_DEF).
 */
template<typename T>
class SubsetArray {
  FORBID_COPY(SubsetArray);

 public:
  /** Element type. */
  typedef T Element;

 private:
  /** Size of an element, in bytes. */
  size_t n_elem_bytes_;
  /** Adjusted beginning of the array. */
  char *adjusted_;
  /** First index. */
  index_t begin_;
  /** One past the last index. */
  index_t end_;

 public:
  SubsetArray() {}
  ~SubsetArray() {
    mem::Free(&(*this)[begin_]);
  }

  /**
   * Initializes this array to a specific range of indices with the specified
   * default element.
   */
  void Init(const Element& default_elem, index_t begin, index_t end);

  /**
   * Gets the size of an element, in bytes.
   */
  index_t n_elem_bytes() const {
    return n_elem_bytes_;
  }

  /**
   * Accesses an element.
   */
  const Element& operator[] (index_t i) const {
    return *reinterpret_cast<Element*>(adjusted_ + i * n_elem_bytes_);
  }

  /**
   * Accesses an element.
   */
  Element& operator[] (index_t i) {
    return *reinterpret_cast<Element*>(adjusted_ + i * n_elem_bytes_);
  }
};

#include "cachearray_impl.h"

#endif

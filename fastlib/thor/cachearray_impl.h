/* Template implementations for cachearray.h. */

template<typename T>
void CacheArrayBlockHandler<T>::Init(const T& default_obj) {
  default_elem_.Init(ot::PointerFrozenSize(default_obj));
  ot::PointerFreeze(default_obj, default_elem_.begin());
}

template<typename T>
void CacheArrayBlockHandler<T>::Serialize(ArrayList<char>* data) const {
  data->Copy(default_elem_);
}

template<typename T>
void CacheArrayBlockHandler<T>::Deserialize(const ArrayList<char>& data) {
  default_elem_.Copy(data);
}

template<typename T>
void CacheArrayBlockHandler<T>::BlockInitFrozen(BlockDevice::blockid_t blockid,
    BlockDevice::offset_t begin, BlockDevice::offset_t bytes, char *block) {
  DEBUG_ASSERT((begin % default_elem_.size()) == 0);
  index_t elems = bytes / default_elem_.size();
  for (index_t i = 0; i < elems; i++) {
    mem::CopyBytes(block, default_elem_.begin(), default_elem_.size());
    block += default_elem_.size();
  }
}

template<typename T>
void CacheArrayBlockHandler<T>::BlockFreeze(BlockDevice::blockid_t blockid,
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

template<typename T>
void CacheArrayBlockHandler<T>::BlockThaw(BlockDevice::blockid_t blockid,
    BlockDevice::offset_t begin, BlockDevice::offset_t bytes,
    char *block) {
  DEBUG_ASSERT(begin % default_elem_.size() == 0);
  index_t elems = bytes / default_elem_.size();
  for (index_t i = 0; i < elems; i++) {
    ot::PointerThaw<T>(block);
    block += default_elem_.size();
  }
}

template<typename T>
void CacheArrayBlockHandler<T>::GetDefaultElement(T *default_element_out) {
  ArrayList<char> tmp(default_elem_);
  const T* source = ot::PointerThaw<T>(tmp.begin());
  ot::Copy(*source, default_element_out);
}

//--------------------------------------------------------------------------

template<typename T>
index_t CacheArray<T>::ConvertBlockSize(
    const Element& element, int kilobytes) {
  size_t elem_size = ot::PointerFrozenSize(element);
  size_t bytes = size_t(kilobytes) << 10;
  int i;

  for (i = 0; (size_t(1) << i) * elem_size <= bytes; i++) {}
  //fprintf(stderr, "%d %d %d %d\n", int(bytes), int(elem_size), int(1 << i), int(elem_size));

  return index_t(1) << (i - 1);
}

template<typename T>
void CacheArray<T>::CreateCacheMaster(int channel,
    index_t n_block_elems, const Element& default_elem, double megs,
    DistributedCache *cache) {
  CacheArrayBlockHandler<Element> *handler =
      new CacheArrayBlockHandler<Element>();
  handler->Init(default_elem);
  cache->InitMaster(channel, n_block_elems * handler->n_elem_bytes(),
      math::RoundInt(megs * MEGABYTE), handler);
}

template<typename T>
void CacheArray<T>::CreateCacheWorker(int channel, double megs,
    DistributedCache *cache) {
  CacheArrayBlockHandler<Element> *handler =
      new CacheArrayBlockHandler<Element>();
  cache->InitWorker(channel, math::RoundInt(megs * MEGABYTE), handler);
}

template<typename T>
void CacheArray<T>::Grow(index_t end_element) {
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

template<typename T>
void CacheArray<T>::Grow() {
  Grow(cache_->n_blocks() << n_block_elems_log());
}

template<typename T>
void CacheArray<T>::Swap(index_t index_a, index_t index_b) {
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

template<typename T>
void CacheArray<T>::Copy(index_t index_src, index_t index_dest) {
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

template<typename T>
index_t CacheArray<T>::AllocD(int owner, index_t count) {
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

template<typename T>
index_t CacheArray<T>::AllocD(int owner) {
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

template<typename T>
void CacheArray<T>::MarkRanges_() {
  if (!BlockDevice::is_dynamic(mode_)) {
    cache_->AddPartialDirtyRange(
        Blockid(begin_), Offset(begin_),
        Blockid(end_), Offset(end_));
  }
}

template<typename T>
void CacheArray<T>::Init(
    DistributedCache *cache_in, BlockDevice::mode_t mode_in,
    index_t begin_index_in, index_t end_index_in) {
  CacheArrayBlockHandler<T>* handler =
      static_cast<CacheArrayBlockHandler<T>*>(
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

template<typename T>
void CacheArray<T>::Flush() {
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

template<typename T>
typename CacheArray<T>::Element* CacheArray<T>::HandleCacheMiss_(
    index_t element_id) {
  BlockDevice::blockid_t victim;
  Metadata *victim_metadata;

  // warning, this isn't very readable... basically, look for the first
  // unlocked item -- the most likely case is that the first item in the
  // fifo is non-negative (i.e. it exists) and it's most likely not locked
  for (;;) {
    if (unlikely(fifo_index_ == 0)) {
      fifo_index_ = FIFO_SIZE;
    }

    fifo_index_--;
    victim = fifo_[fifo_index_];

    if (unlikely(victim < 0)) {
      break;
    }

    victim_metadata = adjusted_metadatas_ + victim;

    if (unlikely(victim_metadata->lock_count != 0)) {
      // the block was locked
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

//--------------------------------------------------------------------------

template<typename Helperclass, typename Element, typename BaseElement>
class ZCacheIterImpl_ {
  FORBID_COPY(ZCacheIterImpl_);

 private:
  Element *element_;
  uint stride_;
  uint left_;
  CacheArray<BaseElement> *cache_;
  BlockDevice::blockid_t blockid_;

 public:
  ZCacheIterImpl_(CacheArray<BaseElement>* cache_in, index_t begin_index) {
    cache_ = cache_in;
    blockid_ = cache_->Blockid(begin_index);
    element_ = Helperclass::MyStartAccess_(cache_, begin_index);
    stride_ = cache_->n_elem_bytes();
    unsigned int mask = cache_->n_block_elems_mask();
    // equivalent to: block_size - (begin_index % block_size) - 1
    left_ = (begin_index ^ mask) & mask;
  }
  ~ZCacheIterImpl_() {
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
class ZCacheReadIterHelperclass_ {
 public:
  static const Element *MyStartAccess_(CacheArray<Element>* a, index_t i) {
    return a->StartRead(i);
  }
};

template<typename Element>
class ZCacheWriteIterHelperclass_ {
 public:
  static Element *MyStartAccess_(CacheArray<Element>* a, index_t i) {
    return a->StartWrite(i);
  }
};

template<typename Helperclass, typename Element, typename BaseElement>
void ZCacheIterImpl_<Helperclass, Element, BaseElement>::NextBlock_() {
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

//--------------------------------------------------------------------------

template<typename T>
void SubsetArray<T>::Init(const Element& default_elem,
    index_t begin, index_t end) {
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

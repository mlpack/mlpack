/*

to-do items:

  - uninitialized pages are only locally initialized when a block is locally
  new - they may be initialized elsewhere.  the "write logic" will help take
  care of this problem.


 */


struct CacheForwardMapping {
  char *data;
  unsigned short lock_count;
  char is_dirty;
};

class CacheArray {
  represents one thread's access to the cache
  acts like an array of elements
  
  what it needs from the cache:
    - give me a block for reading and lock it
    - give me a block for writing and lock it
    - unlock a block for reading
    - unlock a block for writing
};

template<typename T>
class CacheArrayBlockActionHandler {
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

// polymorphic
class BlockDevice {
  FORBID_COPY(SmallCache);

 public:
  typedef uint32 blockid_t;
  typedef uint32 offset_t;

  class SizeListener {
   public:
    virtual void HandleSizeChanged(BlockDevice *device) = 0;
  };

 protected:
  blockid_t n_blocks_;
  offset_t n_block_bytes_;

 private:
  Mutex mutex_;
  ArrayList<SizeListener*> size_listeners_;

 public:
  blockid_t n_blocks() const {
    return n_blocks_;
  }
  offset_t n_block_bytes() const {
    return n_block_bytes_;
  }
  uint64 n_total_bytes() const {
    return uint64(n_blocks_) * n_block_bytes_;
  }

  // todo: rethink the init function
  virtual void Init(datanode *node) = 0;
  virtual void Read(blockid_t blockid, char *data) = 0;
  virtual void Write(blockid_t blockid, const char *data) = 0;

  void AddSizeListener(SizeListener* listener) {
    *size_listeners_.AddBack() = this;
  }

 protected:
  void ChangeSize_(blockid_t n_blocks_new) {
    n_blocks_ = n_blocks_new;
    for (index_t i = 0; i < size_listeners_.size(); i++) {
      size_listeners_[i]->HandleSizeChanged(this);
    }
  }

  void Lock_() { mutex_.Lock(); }
  void Unock_() { mutex_.Unlock(); }
};

class BlockDeviceWrapper : public BlockDevice,
    public BlockDevice::SizeListener {
  FORBID_COPY(BlockDeviceWrapper);
  
 protected:
  BlockDevice *inner_;
  
 public:
  virtual void Init(datanode *datanode) {
    inner_->Init(fx_submodule(datanode, "inner", "inner"));
    n_blocks_ = inner_->n_blocks();
    n_block_bytes_ = inner_->n_block_bytes();
    inner_->AddResizeListener(this);
  }
  
  virtual void Read(blockid_t blockid, char *data) {
    inner_->Read(blockid, data);
  }
  virtual void Write(blockid_t blockid, const char *data) {
    inner_->Write(blockid, data);
  }
  
  void HandleSizeChanged(BlockDevice* device) {
    DEBUG_ASSERT(device == inner_);
    n_blocks_ = inner->n_blocks();
  }
}

/**
 * Extra-simple cache for when everything fits in RAM.
 *
 * All methods here must be locked by a mutex.
 */
template<typename BlockActionHandler>
class SmallCache : public BlockDeviceWrapper {
  FORBID_COPY(SmallCache);
  
 private:
  struct Metadata {
    Metadata() : is_resident(0), is_dirty(0) {}
    char is_resident;
    char is_dirty;
  };
  
 private:
  ArrayList<char> data_;
  ArrayList<Metadata> metadata_;
  BlockActionHandler *handler_;
  
 public:
  char *StartRead(blockid_t block_num);
  char *StartWrite(blockid_t block_num);
  char *StopRead(blockid_t block_num);
  char *StopWrite(blockid_t block_num);
  
  void Init(datanode *datanode) {
    BlockDeviceWrapper::Init(datanode);
    data_.Init(n_total_bytes());
    metadata_.Init(n_blocks());
    handler_ = NULL;
  }
  void Read(blockid_t blockid, char *data);
  void Write(blockid_t blockid, const char *data);
  
  void set_handler(BlockActionHandler *handler_in) {
    handler_ = handler_in;
  }
  
 private:
  char *Lookup_(blockid_t blockid) {
    return n_block_bytes * ptrdiff_t(blockid) + data.begin();
  }
}

char *SmallCache::GetBlock_(blockid_t block_num) {
  if () {
  }
}

class DiskBlockDevice : public BlockDevice {
  FORBID_COPY(DiskBlockDevice);
  
 private:
  int fd_;
  
 public:
  void Read(blockid_t blockid, char *data);
  void Write(blockid_t blockid, const char *data);
};

void DiskBlockDevice::Read(blockid_t blockid, char *data) {
  copy code from blockio to handle incomplete buffers
}

void DiskBlockDevice::Write(blockid_t blockid, const char *data) {
  copy code from blockio to handle incomplete buffers
}

-------------------------------------------------------

template<typename Freezer>
class CacheArray {
  FORBID_COPY(CacheArray);

 public:
  typedef TElement Element;

 private:
  index_t size_;
  index_t live_;

 public:
  CacheArray() {}
  
  ~CacheArray() {
    delete ptr_;
    DEBUG_ONLY(ptr_ = BIG_BAD_NUMBER);
    DEBUG_SAME_INT(live_, 0);
  }
  
  void Init(index_t size_in) {
  }
  
  index_t size() const {
  }

  const Element *StartRead(index_t element_id) {
  }

  void StopRead(const Element *ptr, index_t element_id) {
  }

  Element *StartWrite(index_t element_id) {
  }

  void StopWrite(Element *ptr, index_t element_id) {
  }
  
  void DeclareWritebackRange(index_t start, index_t count) {
  }
  
  void DeclareTempRange(index_t start, index_t count) {
  }
  
  void FlushWritebackRange(index_t start, index_t count) {
  }
};

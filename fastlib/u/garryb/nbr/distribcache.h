
class DistributedCache : public BlockDevice {
 private:

  /** How to query information about the overall state */
  struct DCQueryTransaction {
   public:
    DoneCondition cond;
    Message *response;
   public:
    ~DCQueryTransaction() {}
    
    void Doit(int channel_num);
    void HandleMessage(Message *message);
  };

  /** How to initiate a read messages */
  struct DCReadTransaction {
   public:
    DoneCondition cond;
    Message *response;
   public:
    ~DCReadTransaction() {}
    
    void Doit(int channel_num, int peer, BlockDevice::blockid_t blockid,
        BlockDevice::offset_t begin, BlockDevice::offset_t end,
        char *buffer);
    void HandleMessage(Message *message);
  };

  /** How to initiate a write message */
  struct DCWriteTransaction {
   public:
    void Doit(int channel_num, int peer, BlockDevice::blockid_t blockid,
        BlockDevice::offset_t begin, BlockDevice::offset_t end,
        const char *buffer);
    void HandleMessage(Message *message);
  };

  /** Server-side transaction */
  struct ResponseTransaction : public Transaction {
   public:
    DistributedCache* cache_;

   public:
    void Init(DistributedCache *cache_in);
    void HandleMessage(Message *message);
  };

  /** Server-side channel */
  struct CacheChannel : public Channel {
    DistributedCache *cache;
    Transaction *GetTransaction(Message *message);
  };

 public:
  enum {
    /** The entire block is dirty. */
    FULLY_DIRTY = -1,
    /** Block is not dirty, or end of a dirty-list. */
    NOT_DIRTY_OLD = -2,
    /** Block is new. */
    NOT_DIRTY_NEW = -3
  };

  /**
   * Information we keep about the block.
   *
   * I had to jump through lots of hoops to make this 16 bytes on
   * 64-bit systems.  Don't make me sad by adding anything new to this!
   *
   * See the getters (i.e. is_owner etc) to figure out the actual parameters
   * behave with respect to each other.  Some have much different semantics
   * depending on whether the block is in.
   *
   * The locks variable could still be used for something when the block is
   * out of core.
   */
  struct BlockMetadata {
    BlockMetadata() {}
    ~BlockMetadata() {
      if (data != NULL) {
        mem::Free(data);
      }
    }

    /** Pointer to data, or NULL if not in RAM. */
    char *data;
    /**
     * If negative, the binary complement of who owns it; otherwise,
     * the block ID number.  Note we're using binary complement since
     * unlike negation, ~0 is not 0.
     */
    int32 value;
    /** Linked list of ranges that we've written. */
    int16 dirty_ranges;
    /** Number of locks, or negative if block is new. */
    int16 locks;

    /** Determines the block's owner. */
    int owner(const struct DistributedCache *cache) const {
      return unlikely(value >= 0) ? cache->my_rank_ : (~value);
    }
    /** Determines whether I am the owner. */
    bool is_owner() const {
      return value >= 0;
    }
    /** Determines if block is in RAM. */
    bool is_in_core() const {
      return data != NULL;
    }
    /** Dirtiness check. */
    bool is_dirty() const {
      return dirty_ranges >= FULLY_DIRTY;
    }
    /** Determines whether the block should be treated as blank. */
    bool is_new() const {
      return dirty_ranges == NOT_DIRTY_NEW;
    }
    /** Determines if the block is busy. */
    bool is_busy() const {
      return locks != 0;
    }
    BlockDevice::blockid_t local_blockid() const {
      // A block's ID is invalid if...
      DEBUG_ASSERT_MSG(is_owner(), "no local blockid: it's not mine");
      DEBUG_ASSERT_MSG(!is_in_core(), "no local blockid: it's in RAM!");
      DEBUG_ASSERT_MSG(!is_new(), "no local blockid: it's newly allocated");
      return value;
    }
  };

  /** A network-sendable version of block information. */
  struct BlockStatus {
    uint32 owner;

    OT_DEF(BlockStatus) {
      OT_MY_OBJECT(owner);
    }
  };

  /**
   * External link list of write ranges -- condensed
   * during commits.
   */
  struct RangeLink {
    BlockDevice::offset_t begin;
    BlockDevice::offset_t end;
    uint8 next;
  };
  
  struct OverflowMetadata {
    BlockDevice::blockid_t global_blockid;
    /** Next free local block */
    int32 next_free_local;
  };

 public:
  /* local structures */
  ArrayList<BlockMetadata> blocks_;
  ArrayList<RangeLink> ranges_;
  uint8 free_range_ = -1;
  BlockHandler *handler_;
  
  /* loval device */
  BlockDevice *overflow_device_;
  ArrayList<OverflowMetadata> overflow_metadata_;
  int32 overflow_free_;

  /* remote stuff */
  int channel_num_;
  Channel channel_;
  
  int my_rank_;

 public:
  void Init(int channel_num);

  void Commit();
  void Invalidate();

  char *StartWrite(BlockDevice::blockid_t blockid);
  char *StartWrite(BlockDevice::blockid_t blockid,
      BlockDevice::offset_t begin, BlockDevice::offset_t end);
  char *StartRead(BlockDevice::blockid_t blockid);
  void StopRead(BlockDevice::blockid_t blockid);
  void StopWrite(BlockDevice::blockid_t blockid);

 private:
  void MarkDirty_(BlockMetadata *block,
      BlockDevice::blockid_t begin, BlockDevice::blockid_t end);
  void MarkDirty_(BlockMetadata *block);
  void FreeDirtyList_(BlockMetadata *block);
  void WritebackDirtyRemote_(BlockDevice::blockid_t blockid);
  void WritebackDirtyLocal_(BlockDevice::blockid_t blockid);
};


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
  enum { NOT_DIRTY = -1, FULLY_DIRTY = -2 };

  /** Information we keep about the block. */
  struct BlockMetadata {
    BlockMetadata() {}
    ~BlockMetadata() {
      if (data != NULL) {
        mem::Free(data);
      }
    }

    char *data;
    short dirty_ranges;
    uint8 is_owner;
    uint8 is_new;
    int32 value; // either block's owner or local mapping

    int owner(struct DistributedCache *cache) const {
      return unlikely(is_owner) ? cache->my_rank_ : value;
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

  void AcquireWrite(BlockDevice::blockid_t begin_block,
      BlockDevice::offset_t begin_offset,
      BlockDevice::blockid_t last_block,
      BlockDevice::offset_t end_offset);
  void Commit();
  void Invalidate();
  
 private:
  void MarkDirty_(BlockMetadata *block,
      BlockDevice::blockid_t begin, BlockDevice::blockid_t end);
  void MarkDirty_(BlockMetadata *block);
  void FreeDirtyList_(BlockMetadata *block);
  void WritebackDirtyRemote_(BlockDevice::blockid_t blockid);
  void WritebackDirtyLocal_(BlockDevice::blockid_t blockid);
};

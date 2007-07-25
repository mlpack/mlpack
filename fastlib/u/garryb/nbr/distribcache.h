/*

 Coherency model:

 - Block mapping
   - invalid state exists -- we don't know where the block is
   - state is updated globally at explicit barriers
   - state is updated locally at allocations in two cases
     - 1. I allocate a block and start writing to it -- the block may be
     mapped to myself or may even be mapped to another machine
     - 2. someone else writes a block to me and i didn't realize I was an
     owner
 - Block state
   - Concurrent reads
   - Concurrent writes to even the same block as long as they
   *don't have overlapping ranges*
   - Blocks are write-back to the owner when the block falls out of cache
   - Copies of remote blocks invalidated at barriers
 - Cache
   - Each machine caches nodes.

*/

class DistributedCache : public BlockDevice {
 private:
  /** A network-sendable version of block information. */
  struct BlockStatus {
    uint32 owner;
    bool is_new;

    OT_DEF(BlockStatus) {
      OT_MY_OBJECT(owner);
      OT_MY_OBJECT(is_new);
    }
  };

  struct ConfigResponse {
    BlockDevice::offset_t n_block_bytes;
    ArrayList<char> block_handler_data;

    OT_DEF(ConfigResponse) {
      OT_MY_OBJECT(n_block_bytes);
      OT_MY_OBJECT(block_handler_data);
    }
  };

  struct QueryResponse {
    ArrayList<BlockStatus> statuses;

    OT_DEF(QueryResponse) {
      OT_MY_OBJECT(statuses);
    }
  };

  /** How to query information about the overall state */
  struct ConfigTransaction {
   public:
    DoneCondition cond;
    Message *response;
   public:
    ~ConfigTransaction() { delete response; }

    void Doit(int channel_num, int peer);
    void HandleMessage(Message *message);
  };

  /** How to query information about the overall state */
  struct QueryTransaction {
   public:
    DoneCondition cond;
    Message *response;
   public:
    ~QueryTransaction() { delete response; }

    void Doit(int channel_num, int peer);
    void HandleMessage(Message *message);
  };

  /** How to initiate a read messages */
  struct ReadTransaction {
   public:
    DoneCondition cond;
    Message *response;
   public:
    ~ReadTransaction() {}

    void Doit(int channel_num, int peer, BlockDevice::blockid_t blockid,
        BlockDevice::offset_t begin, BlockDevice::offset_t end,
        char *buffer);
    void HandleMessage(Message *message);
  };

  /** How to initiate a write message */
  struct WriteTransaction {
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

  /** Cache associative entry. */
  struct Slot {
    BlockDevice::blockid_t blockid;

    Slot() { blockid = -1; }
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
  
  enum {
    /** The owner of this block is unknown */
    UNKNOWN_OWNER = -32767,
    /** I'm the owner of this block, but I haven't assigned it a page. */
    SELF_OWNER_UNALLOCATED = 16777216
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
    BlockMetadata() {
      // Initialize into a state where "We know the block is fresh and new
      // but we have no idea who owns it"
      data = NULL;
      value = UNKNOWN_OWNER;
      dirty_ranges = NOT_DIRTY_NEW;
      locks = 0;
    }
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
    /** Determines if the owner of this block is unknown. */
    bool is_unknown() const {
      return value == UNKNOWN_OWNER;
    }
    BlockDevice::blockid_t local_blockid() const {
      // A block's ID is invalid if...
      DEBUG_ASSERT_MSG(is_owner(), "no local blockid: it's not mine");
      DEBUG_ASSERT_MSG(!is_in_core(), "no local blockid: it's in RAM!");
      DEBUG_ASSERT_MSG(!is_new(), "no local blockid: it's newly allocated");
      return value;
    }
  };

  /** An offset range. */
  struct Range {
    offset_t begin;
    offset_t end;
  };

  /**
   * Write ranges.
   */
  struct WriteRanges {
    ArrayList<Range> ranges;
    int next;
  };

  struct OverflowMetadata {
    /** Next free local block */
    BlockDevice::blockid_t next_free_local;
  };

 public:
  /* local structures */
  ArrayList<BlockMetadata> blocks_;
  BlockHandler *handler_;

  ArrayList<WriteRanges> write_ranges_;
  int free_range_;

  /* local device */
  BlockDevice *overflow_device_;
  ArrayList<OverflowMetadata> overflow_metadata_;
  int32 overflow_free_;

  /* remote stuff */
  int channel_num_;
  Channel channel_;

  int my_rank_;
  
  /* cache stuff */
  ArrayList<Slot> slots_;
  unsigned n_sets_;
  int assoc_;
  int log_assoc_;

 public:
  /**
   * Initializes me as the master.
   *
   * The block handler must be valid.
   */
  void InitMaster(int channel_num_in, offset_t n_block_bytes_in,
     size_t total_ram, BlockHandler *initialized_handler_in);
  /**
   * I'm a worker, initialize me.
   *
   * Pass in an unitialized BlockHandler of the desired subclass, and its
   * Deserialize method will be used to initialize it.
   */
  void InitWorker(int channel_num_in,
     size_t total_ram, BlockHandler *initialized_handler_in);
  /** Call this before a rpc::Barrier to initiate synchronization. */
  void PreBarrierSync();
  /** Call this after a rpc::Barrier to complete synchronization. */
  void PostBarrierSync();
  /** Read data as bytes. */
  void Read(blockid_t blockid, offset_t begin, offset_t end, char *buf);
  /** Write data as bytes. */
  void Write(blockid_t blockid, offset_t begin, offset_t end,
      const char *buf);
  /**
   * Writes data from a remote machine.
   *
   * This version of write VERY SPECIFICALLY assumes that if it is receiving
   * a write request, it is AUTOMATICALLY the master.
   */
  void RemoteWrite(blockid_t blockid, offset_t begin, offset_t end,
      const char *buf);

  /* our bread-and-butter cache methods, called by the FIFO */
  char *StartWrite(blockid_t blockid);
  char *StartWrite(blockid_t blockid,
      offset_t begin, offset_t end);
  char *StartRead(blockid_t blockid);
  void StopRead(blockid_t blockid);
  void StopWrite(blockid_t blockid);
 
 private:
  void InitChannel_(int channel_num_in);
  void InitCommon_(offset_t n_block_bytes_);
  void InitCache_(size_t total_ram);
  /**
   * After a sync point, handles the change in ownership info.
   * This is one of the most important parts of our coherency mechanisms.
   */
  void HandleStatusInformation_(const ArrayList<BlockStatus>& statuses);
  /**
   * Tries to grab a block from cache, if it fails, this pulls it from the
   * proper source by calling HandleMiss_.
   */
  void DecacheBlock_(blockid_t blockid);
  /**
   * Puts a block back on the cache after its locks expire.
   */
  void EncacheBlock_(blockid_t blockid);
  /**
   * Grabs a block from the appropriate block device.
   */
  void HandleMiss_(blockid_t blockid);
  /** Grabs a block from our local disk. */
  void HandleLocalMiss_(blockid_t blockid);
  /** Grabs a block from another machine. */
  void HandleRemoteMiss_(blockid_t blockid);
  /**
   * Writes a block back to the appropriate device, if it's dirty -- does
   * nothing if it's not dirty.
   */
  void Writeback_(blockid_t blockid);
  /** Writes back a dirty block to our local disk. */
  void WritebackDirtyLocal_(blockid_t blockid);
  /** Writes back a dirty block to the proper machine. */
  void WritebackDirtyRemote_(blockid_t blockid);
  /** Marks a block as ENTIRELY dirty. */
  void MarkDirty_(BlockMetadata *block);
  /** Marks a block as partially dirty. */
  void MarkDirty_(BlockMetadata *block, offset_t begin, offset_t end);

 private:
  void MarkDirty_(BlockMetadata *block,
      BlockDevice::blockid_t begin, BlockDevice::blockid_t end);
  void MarkDirty_(BlockMetadata *block);
  void FreeDirtyList_(BlockMetadata *block);
  void WritebackDirtyRemote_(BlockDevice::blockid_t blockid);
  void WritebackDirtyLocal_(BlockDevice::blockid_t blockid);
  void HandleStatusInformation_(const ArrayList<BlockStatus>& statuses);
};

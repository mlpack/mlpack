#ifndef THOR_DISTRIBCACHE_H
#define THOR_DISTRIBCACHE_H

#include "col/arraylist.h"
#include "col/rangeset.h"

#include "rpc.h"
#include "cache.h"
#include "blockdev.h"

/*

 Coherency model:

 - Block mapping
   - this section is old so it's probably wrong
   - invalid state exists -- we don't know where the block is
   - state is updated globally at explicit barriers
   - state is updated locally at allocations in two cases
     - 1. I allocate a block and start writing to it -- the block may be
     mapped to myself or may even be mapped to another machine
     - 2. someone else writes a block to me and i didn't realize I was an
     owner
   - blocks can't change mapping once allocated
 - Block state
   - Concurrent reads
   - Concurrent writes to even the same block as long as they
   *don't have overlapping ranges*
   - Blocks are write-back to the owner when the block falls out of cache
   - Copies of remote blocks invalidated at barriers
 - Cache
   - Each machine caches nodes.

FEATURES
 - fast set associative cache
 - handles any identically-sized data structure, even with pointers
 - data distribution
 - as-needed dynamic allocation of disk blocks

FUTURE POSSIBILITIES
- if it's useful for the cache to grow or shrink, it would make little sense
to change the number of sets (that'd be really hard) -- however, it might
make sense to tune the associativity.  i.e. the cache is nominally 16-way
associative, but at runtime you may tune the associativity smaller or larger
depending on memory pressure.

*/

/**
 * A distributed cache that allows overflow onto disk.
 *
 * This is probably the most complicated class that ever existed.  There
 * are lots of helper structures.  They're all really really important.
 * If for some reason the documentation seems to be lacking in the
 * specific details, the ASSERT statements and comments scattered in
 * distribcache.cc should clarify some of the tricky details (such as
 * how a synchronization works).
 */
class DistributedCache : public BlockDevice {
 private:
  /** Net-transferable request operation */
  struct Request {
   public:
    enum { CONFIG, READ, WRITE, ALLOC, OWNER, SYNC } type;
    int32 blockid;
    int32 begin;
    int32 end;
    int32 rank;
    uint64 long_aligned_data_[1];

    template<typename T>
    T *data_as() {
      return reinterpret_cast<T*>(long_aligned_data_);
    }

    static size_t size(size_t data_size) {
      return sizeof(Request) + data_size - sizeof(uint64);
    }
  };

  /** A network-sendable version of block information. */
  struct BlockStatus {
   public:
    int32 owner;
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

  /** How to query information about the overall state */
  struct ConfigTransaction : public Transaction {
   public:
    DoneCondition cond;
    Message *response;
   public:
    ~ConfigTransaction() { delete response; }

    ConfigResponse *Doit(int channel_num, int peer);
    void HandleMessage(Message *message);
  };

  /** How to query information about the overall state */
  struct AllocTransaction : public Transaction {
   public:
    DoneCondition cond;
    Message *response;
   public:
    ~AllocTransaction() { delete response; }

    blockid_t Doit(int channel_num, int peer, blockid_t n_blocks_to_alloc,
        int owner);
    void HandleMessage(Message *message);
  };

  /** How to initiate a read messages */
  struct ReadTransaction : public Transaction {
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
  struct WriteTransaction : public Transaction  {
   public:
    void Doit(int channel_num, int peer, BlockDevice::blockid_t blockid,
        BlockHandler *handler,
        BlockDevice::offset_t begin, BlockDevice::offset_t end,
        const char *buffer);
    void HandleMessage(Message *message);
  };

  /** How to initiate a write message */
  struct OwnerTransaction : public Transaction  {
   public:
    void Doit(int channel_num, int peer,
        BlockDevice::blockid_t blockid, BlockDevice::blockid_t end_block);
    void HandleMessage(Message *message);
  };

  /** Server-side transaction */
  struct ResponseTransaction : public Transaction {
   private:
    DistributedCache* cache_;

   public:
    void Init(DistributedCache *cache_in);
    void HandleMessage(Message *message);
  };

  /** Information that is distributed during a sync. */
  struct SyncInfo {
   public:
    IoStats disk_stats;
    IoStats net_stats;
    ArrayList<BlockStatus> statuses;
    
    OT_DEF(SyncInfo) {
      OT_MY_OBJECT(disk_stats);
      OT_MY_OBJECT(net_stats);
      OT_MY_OBJECT(statuses);
    }
    
   public:
    void Init(const DistributedCache& cache);
    void MergeWith(const SyncInfo& other);
  };

  /**
   * A sync transaction is a barrier to make sure everyone flushes writes,
   * along with a reduction and scatter so each machine knows the updated
   * block owner information.
   */
  struct SyncTransaction : public Transaction  {
   private:
    enum State {
      /** My children and I are flushing data. */
      CHILDREN_FLUSHING,
      /** Others are still flushing. */
      OTHERS_FLUSHING,
      /** My children and I are getting accumulating which blocks we own. */
      CHILDREN_ACCUMULATING,
      /** Parent hasn't yet sent me the authoritative blocks. */
      OTHERS_ACCUMULATING,
      /** Okay, synced and done! */
      DONE
    };

   private:
    DistributedCache *cache_;
    State state_;
    int n_;
    Mutex mutex_;
    SyncInfo sync_info_;

   public:
    void Init(DistributedCache *cache);
    void HandleMessage(Message *message);
    void StartSyncFlushDone();

   private:
    void ChildFlushed_();
    void ParentFlushed_();
    void AccumulateChild_(Message *message);
    void CheckAccumulation_();
    void ParentAccumulated_(const SyncInfo& info);
    void SendBlankSyncMessage_(int peer);
    void SendStatusInformation_(int peer, const SyncInfo& info);
  };

  /** Server-side channel */
  struct CacheChannel : public Channel {
   private:
    DistributedCache *cache_;
    SyncTransaction *sync_transaction_;
    Mutex mutex_;
    DoneCondition sync_done_;

   private:
    SyncTransaction *GetSyncTransaction_();

   public:
    void Init(DistributedCache *cache_in);
    Transaction *GetTransaction(Message *message);
    void SyncDone();
    void StartSyncFlushDone();
    void WaitSync();
  };

  /** Cache associative entry. */
  struct Slot {
    BlockDevice::blockid_t blockid;

    Slot() { blockid = -1; }
  };

 public:
  enum {
    /** The entire block is dirty. */
    FULLY_DIRTY = 0,
    /** Part of the block is dirty, check the ranges. */
    PARTIALLY_DIRTY = 1,
    /** Block is not dirty, or end of a dirty-list. */
    NOT_DIRTY_OLD = 3,
    /** Block is new. */
    NOT_DIRTY_NEW = 5
  };

  enum {
    /** The owner of this block is unknown */
    UNKNOWN_OWNER = -32767,
    /** I'm the owner of this block, but I haven't assigned it a page. */
    SELF_OWNER_UNALLOCATED = (1 << 30)
  };

  /**
   * Rank of the master machine, which is configured with the block handler
   * and the number of bytes in a block, and maybe other setup parameters
   * (this is the destination of ConfigTransaction).
   */
  static const int MASTER_RANK = 0;

  /** Log of the set associativity, i.e. 3 means 2^3 = 8-way */
  static const int LOG_ASSOC = 4;
  /** The set associativity of the cache. */
  static const int ASSOC = (1 << LOG_ASSOC);

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
      DEBUG_ASSERT_MSG(sizeof(BlockMetadata) <= 16,
          "Don't make the metadata take too much RAM!");
      data = NULL;
      value = UNKNOWN_OWNER;
      locks = 0;
      status = NOT_DIRTY_NEW;
      is_reading = false;
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
    /** Number of FIFO's that are currently accessing this. */
    int16 locks;
    /** Linked list of ranges that we've written. */
    uint8 status;
    /** Whether the block is currently being read asynchronously. */
    uint8 is_reading;

    /** Determines the block's owner. */
    int owner(const struct DistributedCache *cache) const {
      return unlikely(value >= 0) ? cache->my_rank_ : (~value);
    }
    int owner() const {
      DEBUG_ASSERT_MSG(!is_owner(), "owner() doesn't work if i'm the owner");
      return ~value;
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
      return status <= PARTIALLY_DIRTY;
    }
    /** Determines whether the block should be treated as blank. */
    bool is_new() const {
      return status == NOT_DIRTY_NEW;
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
      DEBUG_ASSERT_MSG(!is_new(), "no local blockid: it's newly allocated");
      return value;
    }
  };

  struct Position {
   public:
    blockid_t block;
    offset_t offset;
    
    OT_DEF(Position) {
      OT_MY_OBJECT(block);
      OT_MY_OBJECT(offset);
    }

   public:
    bool operator < (const Position& other) const {
      if (unlikely(block == other.block)) {
        return offset < other.offset;
      } else {
        return block < other.block;
      }
    }
    bool operator == (const Position& other) const {
      return block == other.block && offset == other.offset;
    }
    DEFINE_ALL_COMPARATORS(Position);
  };

 public:
  /* local structures */
  ArrayList<BlockMetadata> blocks_;
  BlockHandler *handler_;

  /* ranges that apply for partial writes */
  RangeSet<Position> write_ranges_;

  /* local device */
  BlockDevice *overflow_device_;
  DenseIntMap<blockid_t> overflow_next_;
  int32 overflow_free_;

  /* remote stuff */
  int channel_num_;
  CacheChannel channel_;

  int my_rank_;
  
  /* cache stuff */
  ArrayList<Slot> slots_;
  unsigned n_sets_;

  Mutex mutex_;
  WaitCondition io_cond_;
  
  IoStats disk_stats_;
  IoStats net_stats_;
  IoStats world_disk_stats_;
  IoStats world_net_stats_;

 public:
  DistributedCache() {}
  virtual ~DistributedCache();

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
  /**
   * Flushes all dirty blocks that live on other machines, but leaves
   * them in cache.
   *
   * It may be possible for a performance benefit from calling this
   * periodically -- maybe mostly due to the ability to avoid lots of
   * extra reads.
   *
   * @param portion the portion (out of 1.0) to attempt flushing, in case
   *        a "soft flush" is desired -- i.e. a flush of 0.5 will flush out
   *        only old blocks
   */
  void BestEffortWriteback(double portion = 1.0);
  /** Starts syncing. */
  void StartSync();
  /**
   * Ensures that the sync point has passed before returning.
   *
   * At this point, you can optionally store I/O statistics in a module.
   * Local I/O statistics are cleared after they are stored!  However, we
   * don't clear world statistics until the next sync.  If you want your disk
   * stats to be 100% accurate you need to place a barrier right after all
   * of your WaitSyncs before other machines start writing.
   */
  void WaitSync(datanode *node = NULL);
  /**
   * Synchronizes status with all machines.
   *
   * What this does is writes back all blocks (StartSync), and then in
   * a tree-like manner, reconciles all the block ownership information.
   *
   * To hide the sync latency somewhat, you can StartSync several
   * synchronizations, optionally do some stuff in between, and
   * then StopSync all of the synchronizations instead.
   */
  void Sync() {
    StartSync();
    WaitSync();
  }
  /**
   * Sets the entire cache to the default values.
   */
  void ResetElements();
  /** Read data as bytes. */
  void Read(blockid_t blockid, offset_t begin, offset_t end, char *buf);
  /** Write data as bytes. */
  void Write(blockid_t blockid, offset_t begin, offset_t end,
      const char *buf);
  /**
   * Read data as bytes for a remote machine.
   *
   * Currently, this differs from Read only that it fails out if we're not
   * the block's actual owner.
   */
  void RemoteRead(blockid_t blockid, offset_t begin, offset_t end, char *buf);
  /**
   * Writes data from a remote machine.
   *
   * This version of write assumes that if it is receiving
   * a write request, it is now the block's owner.
   */
  void RemoteWrite(blockid_t blockid, offset_t begin, offset_t end,
      const char *buf);

  /* our bread-and-butter cache methods, called by the FIFO */
  /** Start a write access to the whole page. */
  char *StartWrite(blockid_t blockid, bool is_partial);
  /** Start a read to a page. */
  char *StartRead(blockid_t blockid);
  /** End a read access. */
  void StopRead(blockid_t blockid);
  /** End a write access. */
  void StopWrite(blockid_t blockid);
  /**
   * Adds a partial dirty range.
   *
   * The only way to remove a dirty range is by a sync barrier.
   */
  void AddPartialDirtyRange(blockid_t begin_block, offset_t begin_offset,
      blockid_t last_block, offset_t end_offset);
  /**
   * Gives ownership of my block to a new owner.
   *
   * What this does is resets the block's owner to the new owner and marks
   * it as fully dirty, so a flush or synchronization will complete the
   * ownership transfer.
   */
  void GiveOwnership(blockid_t my_block, int new_owner);

  /** Allocates blocks, becoming myself the owner of these blocks. */
  blockid_t AllocBlocks(blockid_t n_blocks_to_alloc) {
    return AllocBlocks(n_blocks_to_alloc, my_rank_);
  }
  /** Allocates blocks, but assign ownership to a specified machine. */
  blockid_t AllocBlocks(blockid_t n_blocks_to_alloc, int owner);
  /** Backend for AllocBlocks. */
  blockid_t RemoteAllocBlocks(
      blockid_t n_blocks_to_alloc, int owner, int sender);
  /** Gets the underlying block handler. */
  BlockHandler *block_handler() const {
    return handler_;
  }

  const IoStats& disk_stats() const {
    return disk_stats_;
  }
  const IoStats& net_stats() const {
    return net_stats_;
  }
  const IoStats& world_disk_stats() const {
    return world_disk_stats_;
  }
  const IoStats& world_net_stats() const {
    return world_net_stats_;
  }
  int channel_num() const {
    return channel_num_;
  }

 private:
  void InitChannel_(int channel_num_in);
  void InitCommon_();
  void InitCache_(size_t total_ram);
  void InitFile_(const char *fname);
  /**
   * After a sync point, handles the change in ownership info.
   * This is one of the most important parts of our coherency mechanisms.
   */
  void HandleStatusInformation_(const ArrayList<BlockStatus>& statuses);
  /**
   * After a sync point, handles synchronization information.
   */
  void HandleSyncInfo_(const SyncInfo& info);
  /**
   * Marks me as owner of blocks I own and marks other blocks as null.
   */
  void ComputeStatusInformation_(ArrayList<BlockStatus>* statuses) const;
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
   * Removes a block's memory, if it's dirty -- does
   * nothing if it's not dirty.
   */
  void Purge_(blockid_t blockid);
  /**
   * Writes back a dirty block to our local disk.
   * Resets the status to clean -- although this does not clear the data, this
   * will freeze the data!
   */
  void WritebackDirtyLocalFreeze_(blockid_t blockid);
  /**
   * Writes back a dirty block to the proper machine.
   * Resets status to clean but does not clear the data.
   */
  void WritebackDirtyRemote_(blockid_t blockid);
  /** Marks a block as ENTIRELY dirty. */
  void MarkDirty_(BlockMetadata *block);
  /** Marks a block as partially dirty. */
  void MarkDirty_(BlockMetadata *block, offset_t begin, offset_t end);
  /** Mark a particular owner for a region of blocks. */
  void MarkOwner_(int owner, blockid_t begin, blockid_t end);
  /** Handle the fact that I'm suddenly the owner of these blocks. */
  void HandleRemoteOwner_(blockid_t block, blockid_t end);
};

#endif

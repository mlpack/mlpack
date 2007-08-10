#ifndef THOR_DISTRIBCACHE_H
#define THOR_DISTRIBCACHE_H

#include "col/arraylist.h"
#include "col/rangeset.h"

#include "rpc.h"
#include "cache.h"
#include "blockdev.h"

/*
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
 * This is probably the most complicated class in THOR or FASTlib.  There
 * are lots of helper structures.  They're all really really important.  If
 * for some reason the documentation seems to be lacking in the specific
 * details, the ASSERT statements and comments scattered in distribcache.cc
 * should clarify some of the tricky details (such as how a synchronization
 * works).
 *
 * The API of this class is about "checking out" blocks for reading, writing,
 * etc.  When a block is checked out, if the block is already checked out
 * by another process, the lock count is only increased.  If not, first the
 * cache is searched, otherwise it is obtained from another machine or from
 * disk.  When writes are performed, they may be to the entire block
 * (a "fully dirty" block) or they may be marked as "partial" (to a
 * contiguous region).  Contiguous regions are marked as "partially
 * dirty ranges" to the entire array.
 *
 * A lot of the code is dedicated to the protocol.  The different kinds
 * of messages are documented at the class level inside.
 *
 * Consistency model: Data that is being written to is only consistent if
 * you are the thread that is writing the data, or if a global sync point
 * has been reached.  Concurrent reads and writes are allowed to the same
 * block as long as they are in disjoint ranges.  See above for the
 * distinction between "fully dirty" (such as a block that has been
 * allocated, and is not subject to any masking by partially dirty ranges)
 * and "partially dirty", to which writes are constrained by a set of
 * contiguous regions.
 */
class DistributedCache : public BlockDevice {
 private:
  /** Net-transferable request operation */
  struct Request {
   public:
    /** Message type.  See corresponding transaction classes. */
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

  /**
   * A network-sendable version of block information.
   *
   * In a synchronization event, each machine indicates status about all
   * blocks it owns.  The only piece of information needed other than the
   * owner is whether the block is new.
   */
  struct BlockStatus {
   public:
    /** Indicates who is the owner of this block. */
    int32 owner;
    /**
     * Indicates if the block's contents have been set to anything
     * other than the initial value -- this can save unnecessary
     * READs in many cases.  For instance, the "results" array usually
     * follows this behavior.
     */
    bool is_new;

    OT_DEF_BASIC(BlockStatus) {
      OT_MY_OBJECT(owner);
      OT_MY_OBJECT(is_new);
    }
  };

  /**
   * Response to a configuration request.
   *
   * This contains all information needed for workers to initialize their
   * distributed cache.
   */
  struct ConfigResponse {
    /**
     * The block size, in bytes.
     */
    BlockDevice::offset_t n_block_bytes;
    /**
     * Information the block handler (or "schema") needs to initialize itself
     * with.
     */
    ArrayList<char> block_handler_data;

    OT_DEF(ConfigResponse) {
      OT_MY_OBJECT(n_block_bytes);
      OT_MY_OBJECT(block_handler_data);
    }
  };

  /**
   * The transaction used on the listening side to respond to any
   * incoming messages.
   */
  struct ResponseTransaction : public Transaction {
   private:
    DistributedCache* cache_;

   public:
    void Init(DistributedCache *cache_in);
    void HandleMessage(Message *message);
  };

  /**
   * Information that is transmitted to all machines during a
   * synchronization point.
   */
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
   *
   * In this, in the first step, all machines make sure that all of the
   * others have finished writing and reached the synchronization point,
   * like in a barrier.  Next, all machines relay in a tree structure to
   * the root (like a reduction) the blocks that they own and the status,
   * and the root broadcasts to all machines (in a tree topology) .
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

  /** Server-side channel, listening for asynchronous requests. */
  struct CacheChannel : public Channel {
   private:
    /** The distributed cache backing this. */
    DistributedCache *cache_;
    /** The synchronization event, non-null if one is occurring. */
    SyncTransaction *sync_transaction_;
    /** A mutual exclusion lock */
    Mutex mutex_;
    /** Condition indicating when the current sync is done. */
    DoneCondition sync_done_;

   private:
    /** Gets the current sync transaction, creating one if necessary. */
    SyncTransaction *GetSyncTransaction_();

   public:
    /** Initializes this channel to listen to message for this cache. */
    void Init(DistributedCache *cache_in);
    /**
     * Returns the proper transaction for the message, either a new
     * ResponseTransaction or the current sync transaction.
     */
    Transaction *GetTransaction(Message *message);
    /**
     * Called by SyncTransaction, indicates that syncing is done.
     * (The wait condition doesn't exist in SyncTransaction since
     * SyncTransaction must promptly delete itself to avoid race
     * conditions.)
     */
    void SyncDone();
    /**
     * Locally indicates that synchronization should start, that we have
     * already flushed all our data and emptied all the write queues.
     */
    void StartSyncFlushDone();
    /** Waits until synchronization is finished. */
    void WaitSync();
  };

  /** An entry in the set-associative LRU cache. */
  struct Slot {
    /**
     * The block ID associated with this entry, or -1 if this slot is
     * currently unused.
     */
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
   * (this is the destination of the CONFIG message).
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
  /** A block-to-metadata mapping, keeping track of each block's status. */
  ArrayList<BlockMetadata> blocks_;
  /** A schema that regards how to freeze, thaw, and initialize blocks. */
  BlockHandler *handler_;

  /** Mutual exclusion lock to serialize all metadata access. */
  Mutex mutex_;
  
  /** The associtive cache. */
  ArrayList<Slot> slots_;
  /** The number of cache sets, same as slots_.size() / ASSOC. */
  unsigned n_sets_;

  /** The rank of the current machine, same as rpc::rank(). */
  int my_rank_;

  /** Contiguous regions that are marked as dirty. */
  RangeSet<Position> write_ranges_;

  /** A larger device to overflow blocks if there isn't enough room. */
  BlockDevice *overflow_device_;
  /** Maintains a freelist of local blocks. */
  DenseIntMap<blockid_t> overflow_next_;
  /** The head of the freelist for local blocks. */
  int32 overflow_free_;

  /** The channel number associated with this cache. */
  int channel_num_;
  /** The channel listening for remote requests. */
  CacheChannel channel_;
  /** Wait condition used to listen for responses to read requests. */
  WaitCondition io_cond_;

  /** I/O stats for our own disk. */
  IoStats disk_stats_;
  /** I/O stats for our own network requests. */
  IoStats net_stats_;
  /** I/O stats for all disks. */
  IoStats world_disk_stats_;
  /** I/O stats for all network requests. */
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
   * At this point, you can optionally report I/O statistics to a module.
   * Local I/O statistics are cleared after they are stored!  However, we
   * don't clear world statistics until the next sync.  If you want your disk
   * stats to be 100% accurate you need to place a barrier right after all
   * of your WaitSyncs before other machines start writing.
   *
   * Food for thought: Currently this doesn't report local stats (because
   * they're typically not very interesting), and since remote stats are
   * not cleared, it might make more sense to have a separate function
   * that just reports world stats.  However, a sync point is a nice natural
   * place to put this anyway.
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
  /** The back-end for AllocBlocks. */
  blockid_t RemoteAllocBlocks(
      blockid_t n_blocks_to_alloc, int owner, int sender);
  /** Gets the underlying block handler. */
  BlockHandler *block_handler() const {
    return handler_;
  }

  /** IO statistics for local disk. */
  const IoStats& disk_stats() const {
    return disk_stats_;
  }
  /** IO statistics for my machine accessing remote machines. */
  const IoStats& net_stats() const {
    return net_stats_;
  }
  /** IO statistics for all machines using disk. */
  const IoStats& world_disk_stats() const {
    return world_disk_stats_;
  }
  /** IO statistics for current machine using disk. */
  const IoStats& world_net_stats() const {
    return world_net_stats_;
  }
  /** The channel number associated with this cache. */
  int channel_num() const {
    return channel_num_;
  }

 private:
  /** Sets up the channel to listen for remote requests. */
  void InitChannel_(int channel_num_in);
  /** Initializes things in common for masters and workers. */
  void InitCommon_();
  /** Creates the associative cache array. */
  void InitCache_(size_t total_ram);
  /** Opens up the overflow file. */
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
  /** Recycles a local block ID for use later. */
  void RecycleLocalBlock_(blockid_t blockid);	

  /**
   * Requests configuration from the master.
   *
   * This determines the number of bytes in a block, and the contents
   * of the default initial element (i.e. the information the
   * BlockHandler needs to do its job).
   */
  void DoConfigRequest_();
  /**
   * Requests a block allocation from the master.
   *
   * This just sends a message to the master asking for a block and responds
   * with a free block.
   */
  blockid_t DoAllocRequest_(blockid_t n_blocks_to_alloc, int owner);
  /**
   * Informs a peer that it has become the owner of a block.
   *
   * Whenever we allocate a block but declare someone else the owner, we
   * must inform them that they are the owner, so that in a synchronization
   * point, the remote host will correctly be able to identify the block
   * as their own.
   */
  void DoOwnerRequest_(int owner, blockid_t blockid, blockid_t end_block);
  /**
   * Sends out a write to be executed.
   *
   * This has to freeze the data inside to avoid mucking up the data that's
   * stored in RAM.
   */
  void DoWriteRequest_(int peer, blockid_t blockid,
      offset_t begin, offset_t end, const char *buffer);
  /**
   * Sends a read request and reads the data into the buffer.
   *
   * Doesn't thaw the data, leaves it up to the caller.
   */
  void DoReadRequest_(int peer, blockid_t blockid,
      offset_t begin, offset_t end, char *buffer);
};

#endif

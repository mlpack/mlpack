#ifndef NBR_CACHE_H
#define NBR_CACHE_H

#include "blockdev.h"

/**
 * Handles events associated with a cache pulling blocks in and out of memory.
 */
class BlockHandler {
  FORBID_COPY(BlockHandler);
 public:
  BlockHandler() {}
  virtual ~BlockHandler() {}

  /** Save state. */
  virtual void Serialize(ArrayList<char>* data) const = 0;
  /** Initialize from state. */
  virtual void Deserialize(const ArrayList<char>& data) = 0;
  /** Initialize a chunk in frozen state. */
  virtual void BlockInitFrozen(BlockDevice::blockid_t blockid,
      BlockDevice::offset_t begin,
      BlockDevice::offset_t bytes, char *block) = 0;
  /** Freeze a chunk so it can be sent over net, or stored. */
  virtual void BlockFreeze(BlockDevice::blockid_t blockid,
      BlockDevice::offset_t begin, BlockDevice::offset_t bytes,
      const char *old_location, char *block) = 0;
  /** Thaw a chunk so it can be accessed via pointers. */
  virtual void BlockThaw(BlockDevice::blockid_t blockid,
      BlockDevice::offset_t begin, BlockDevice::offset_t bytes,
      char *block) = 0;
};

#if 0
/**
 * Extra-simple cache for when everything fits in RAM.
 *
 * All methods here must be locked by a mutex.
 */
class SmallCache : public BlockDeviceWrapper {
  FORBID_COPY(SmallCache);

 private:
  struct Metadata {
    Metadata() : data(NULL), lock_count(0) { }
    /** current version of the data, can be NULL */
    char *data;
    int lock_count;
  };

 private:
  static const BlockDevice::blockid_t HEADER_BLOCKID = 0;

 private:
  Mutex mutex_;
  ArrayList<Metadata> metadatas_;
  mode_t mode_;
  BlockHandler *block_handler_;

 public:
  SmallCache() {
    DEBUG_ONLY(n_blocks_ = BIG_BAD_NUMBER);
  }
  virtual ~SmallCache();

  mode_t mode() const {
    return mode_;
  }
  
  BlockHandler *block_handler() const {
    return block_handler_;
  }

  /**
   * Create a SmallCache.
   */
  void Init(BlockDevice *inner_in, BlockHandler *block_handler_in,
     mode_t mode_in);

  char *StartRead(blockid_t blockid);
  char *StartWrite(blockid_t blockid);
  void StopRead(blockid_t blockid);
  void StopWrite(blockid_t blockid);

  /**
   * Flush method for static (range-based) use cases.
   *
   * Flushes [ begin_block:begin_offset , last_block:endoffset )
   *
   * @param clear free up all blocks that are no longer in use -- this is
   *        DANGEROUS when used improperly -- only use when you are sure
   *        any active block is marked as "locked", i.e. the CacheArray
   *        class with no replacement policy would be absolutely fine
   */
  void Flush(bool clear, blockid_t begin_block, offset_t begin_offset,
      blockid_t last_block, offset_t end_offset);
  /**
   * Flush method for dynamic use cases.
   *
   * Flushes all contents of all blocks that are currently active.
   * Optionally removes them from the cache.
   *
   * Flushes [ begin_block:begin_offset , last_block:endoffset )
   *
   * @param cache whether to remove from cache
   */
  void Flush(bool clear);
  /**
   * Frees up all blocks.
   */
  void Clear();
  /**
   * Changes mode.
   */
  void Remode(mode_t new_mode);

  virtual void Read(blockid_t blockid,
      offset_t begin, offset_t end, char *data);
  virtual void Write(blockid_t blockid,
      offset_t begin, offset_t end, const char *data);

  virtual blockid_t AllocBlocks(blockid_t n_to_alloc);

 private:
  void PerformCacheMiss_(blockid_t blockid);
  void Writeback_(bool clear,
      blockid_t blockid, offset_t begin, offset_t end);

  Metadata *GetBlock_(blockid_t blockid);
};
#endif

#endif

#ifndef NBR_CACHE_H
#define NBR_CACHE_H

#include "blockdev.h"

/**
 * Extra-simple cache for when everything fits in RAM.
 *
 * All methods here must be locked by a mutex.
 */
class SmallCache : public BlockDeviceWrapper {
  FORBID_COPY(SmallCache);

 private:
  struct Metadata {
    Metadata() : data(NULL), lock_count(0) {}
    /** current version of the data, can be NULL */
    char *data;
    int lock_count;
  };
  
 private:
  static const BlockDevice::blockid_t HEADER_BLOCKID = 0;

 private:
  Mutex mutex_;
  ArrayList<Metadata> metadatas_;
  Schema *schema_;
  mode_t mode_;

 public:
  SmallCache() {
    DEBUG_ONLY(n_blocks_ = BIG_BAD_NUMBER);
  }
  virtual ~SmallCache();

  mode_t mode() const {
    return mode_;
  }

  /**
   * Create a SmallCache.
   */
  void Init(BlockDevice *inner_in, Schema *schema_in,
     mode_t mode_in);

  char *StartRead(blockid_t blockid);
  char *StartWrite(blockid_t blockid);
  void StopRead(blockid_t blockid);
  void StopWrite(blockid_t blockid);

  /**
   * Flush method for static (range-based) use cases.
   */
  void Flush(blockid_t begin_block, offset_t begin_offset,
      blockid_t last_block, offset_t end_offset);
  /**
   * Flush method for dynamic use cases.
   */
  void Flush();
  /**
   * Invalidates all current blocks, requiring them to be fetched from the
   * underlying block device.
   *
   * Any un-flushed writes will be lost -- be careful that blocks aren't
   * left partly-uninitialized.
   *
   * Useful in parallel usage where parts of the underlying data might have
   * changed elsewhere.
   *
   * TODO: Consider marking some blocks as "invalidate candidates" so we
   * only need to invalidate blocks at range boundaries.
   */
  void Clear(mode_t new_mode);
  /**
   * Change mode without invalidating.
   *
   * Useful 
   */
  void Remode(mode_t new_mode);

  virtual void Read(blockid_t blockid,
      offset_t begin, offset_t end, char *data);
  virtual void Write(blockid_t blockid,
      offset_t begin, offset_t end, const char *data);

  virtual blockid_t AllocBlocks(blockid_t n_to_alloc) {
    blockid_t blockid = BlockDeviceWrapper::AllocBlock();
    metadatas_.Resize(n_blocks());
    return blockid;
  }

 private:
  void PerformCacheMiss_(blockid_t blockid);
  void Writeback_(blockid_t blockid, offset_t begin, offset_t end);

  Metadata *GetBlock_(blockid_t blockid) {
    if (unlikely(blockid >= n_blocks_)) {
      PerformCacheMiss_(blockid);
    }

    Metadata *metadata = &metadatas_[blockid];
    char *data = metadata->data;

    if (unlikely(data == NULL)) {
      PerformCacheMiss_(blockid);
    }

    return metadata;
  }
};

#endif

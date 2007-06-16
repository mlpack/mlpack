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
  Schema *handler_;
  mode_t mode_;

 public:
  SmallCache() {}
  virtual ~SmallCache();

  mode_t mode() const {
    return mode_;
  }

  /**
   * Create a SmallCache.
   *
   * If mode is READ or MODIFY, the incoming handler's InitFromMetadata method
   * will be called.
   *
   * If the mode is WRITE or TEMP, the handler is assumed to be initialized,
   * and its WriteMetadata will be called.
   */
  void Init(BlockDevice *inner_in, Schema *handler_in,
     mode_t mode_in);

  char *StartRead(blockid_t blockid);
  char *StartWrite(blockid_t blockid);
  void StopRead(blockid_t blockid);
  void StopWrite(blockid_t blockid);
  void Flush(blockid_t begin_block, offset_t begin_offset,
    blockid_t last_block, offset_t end_offset);

  virtual void Read(blockid_t blockid,
      offset_t begin, offset_t end, char *data);
  virtual void Write(blockid_t blockid,
      offset_t begin, offset_t end, const char *data);

  virtual blockid_t AllocBlock() {
    blockid_t blockid = BlockDeviceWrapper::AllocBlock();
    metadatas_.Resize(n_blocks());
    return blockid;
  }
  
  void Clear(mode_t new_mode);

  Schema *block_action_handler() const {
    return handler_;
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

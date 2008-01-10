/**
 * @file cache.h
 *
 * Common abstractions used by the caching infrastructure.
 */

#ifndef THOR_CACHE_H
#define THOR_CACHE_H

#include "blockdev.h"

/**
 * Handles events associated with a cache pulling blocks in and out of memory.
 */
class BlockHandler {
  FORBID_ACCIDENTAL_COPIES(BlockHandler);
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

#endif

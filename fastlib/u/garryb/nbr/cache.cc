#include "cache.h"

void SmallCache::Init(
    BlockDevice *inner_in, BlockHandler *block_handler_in, mode_t mode_in) {
  BlockDeviceWrapper::Init(inner_in); // sets inner_, n_block_bytes_, etc

  metadatas_.Init(n_blocks_);
  block_handler_ = block_handler_in;
  mode_ = mode_in;
}

SmallCache::~SmallCache() {
  for (index_t i = 0; i < metadatas_.size(); i++) {
    Metadata *metadata = &metadatas_[i];
    mem::Free(metadata->data);
    DEBUG_SAME_INT(metadatas_[i].lock_count, 0);
    DEBUG_POISON_PTR(metadata->data);
  }
  delete block_handler_;
}

void SmallCache::Clear() {
  for (index_t i = 0; i < metadatas_.size(); i++) {
    Metadata *metadata = &metadatas_[i];
    DEBUG_SAME_INT(metadata->lock_count, 0);
    mem::Free(metadata->data);
    metadata->data = NULL;
    metadata->lock_count = 0;
  }
}

void SmallCache::Remode(mode_t new_mode) {
  // TODO: Assert that no blocks are open
  mode_ = new_mode;
}

char *SmallCache::StartRead(blockid_t blockid) {
  Lock();

  Metadata *metadata = GetBlock_(blockid);
  metadata->lock_count++;

  Unlock();

  return metadata->data;
}

char *SmallCache::StartWrite(blockid_t blockid) {
  Lock();

  Metadata *metadata = GetBlock_(blockid);
  metadata->lock_count++;
  
  Unlock();

  return metadata->data;
}

void SmallCache::StopRead(blockid_t blockid) {
  Lock();
  Metadata *metadata = GetBlock_(blockid);
  --metadata->lock_count;
  Unlock();
}

void SmallCache::StopWrite(blockid_t blockid) {
  Lock();
  Metadata *metadata = GetBlock_(blockid);
  --metadata->lock_count;
  Unlock();
}

SmallCache::Metadata *SmallCache::GetBlock_(blockid_t blockid) {
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

void SmallCache::PerformCacheMiss_(blockid_t blockid) {
  char *data;
  Metadata *metadata;

  if (unlikely(blockid >= n_blocks_)) {
    n_blocks_ = blockid + 1;
    metadatas_.Resize(n_blocks_);
  }

  metadata = &metadatas_[blockid];

  data = mem::Alloc<char>(n_block_bytes());
  if (BlockDevice::need_read(mode_)) {
    inner_->Read(blockid, data);
    block_handler_->BlockThaw(blockid, 0, n_block_bytes_, data);
  } else {
    // If a replacement policy were used, we'd have to make sure the block
    // hadn't been written back previously.
    block_handler_->BlockInitFrozen(blockid, 0, n_block_bytes_, data);
    block_handler_->BlockThaw(blockid, 0, n_block_bytes_, data);
  }
  metadata->data = data;
}

void SmallCache::Writeback_(bool clear,
    blockid_t blockid, offset_t begin, offset_t end) {
  if (likely(begin != end) && likely(blockid < n_blocks_)) {
    Metadata *metadata = &metadatas_[blockid];
    char *data = metadata->data;

    DEBUG_BOUNDS(begin, end + 1);

    if (data) {
      size_t n_bytes = end - begin;
      char *buf = data + begin;

      block_handler_->BlockFreeze(blockid, 0, n_bytes, buf, buf);
      inner_->Write(blockid, begin, end, buf);

      if (clear && metadata->lock_count == 0) {
        // block is not in use, get rid of it
        mem::Free(metadata->data);
        metadata->data = NULL;
      } else {
        // we must re-thaw the contents
        block_handler_->BlockThaw(blockid, 0, n_bytes, buf);
      }
    }
  }
}

void SmallCache::Flush(bool clear,
    blockid_t begin_block, offset_t begin_offset,
    blockid_t last_block, offset_t end_offset) {
  DEBUG_ASSERT(BlockDevice::can_write(mode_));

  if (BlockDevice::need_write(mode_)) {
    if (begin_block == last_block) {
      Writeback_(clear, begin_block, begin_offset, end_offset);
    } else {
      Writeback_(clear, begin_block, begin_offset, n_block_bytes_);
      for (blockid_t i = begin_block + 1; i < last_block; i++) {
        Writeback_(clear, i, 0, n_block_bytes_);
      }
      Writeback_(clear, last_block, 0, end_offset);
    }
  }
}

void SmallCache::Flush(bool clear) {
  DEBUG_ASSERT(BlockDevice::can_write(mode_));

  if (BlockDevice::need_write(mode_)) {
    for (blockid_t i = 0; i < n_blocks_; i++) {
      Writeback_(clear, i, 0, n_block_bytes_);
    }
  }
}

void SmallCache::Read(blockid_t blockid,
    offset_t begin, offset_t end, char *buf) {
  const char *src = StartRead(blockid);
  size_t n_bytes = end - begin;

  // TODO: consider read-through
  mem::Copy(buf, src + begin, n_bytes);
  block_handler_->BlockFreeze(blockid, begin, n_bytes, src + begin, buf);

  StopRead(blockid);
}

void SmallCache::Write(blockid_t blockid,
    offset_t begin, offset_t end, const char *buf) {
  // TODO: Consider write-through on straddles (avoiding the read and
  // cache pollution)
  // TODO: Consider skipping the read when entire block is written (though
  // still may consider keeping block in cache)
  // TODO: This behavior may be necessary due to the specious behavior of
  // Flush et al.  It's assumed though anyone using this as a block device
  // is well aware of the behavior under Flush and is willing to live with
  // the consequences -- i.e. this would probably have to be the ONLY
  // cache accessing the underlying block device.

  char *dest = StartWrite(blockid);
  size_t n_bytes = end - begin;

  mem::CopyBytes(dest + begin, buf, n_bytes);
  block_handler_->BlockThaw(blockid, begin, n_bytes, dest + begin);

  StopWrite(blockid);
}

BlockDevice::blockid_t SmallCache::AllocBlocks(blockid_t n_to_alloc) {
  Lock();
  blockid_t blockid = BlockDeviceWrapper::AllocBlocks(n_to_alloc);
  metadatas_.Resize(n_blocks());
  Unlock();
  //fprintf(stderr, "Alloc %d, get %d, n_blocks %d\n", n_to_alloc, blockid, n_blocks());

  return blockid;
}

#include "cache.h"

void SmallCache::Init(
    BlockDevice *inner_in, Schema *schema_in, mode_t mode_in) {
  BlockDeviceWrapper::Init(inner_in); // sets inner_, n_block_bytes_, etc

  metadatas_.Init(n_blocks_);
  schema_ = schema_in;
  mode_ = mode_in;
/*
  ArrayList<char> header_data;
  header_data.Init(inner_->n_block_bytes());

  if (mode_ == CREATE || mode_ == TEMP) {
    char *data = StartWrite(HEADER_BLOCKID);
    schema_->WriteHeader(n_block_bytes_, data);
    StopWrite(HEADER_BLOCKID);
  } else if (mode_ == READ || mode_ == MODIFY) {
    char *data = StartRead(HEADER_BLOCKID);
    schema_->InitFromHeader(n_block_bytes_, data);
    StopRead(HEADER_BLOCKID);
  } else {
    FATAL("Unknown mode");
  }
*/
}

SmallCache::~SmallCache() {
  for (index_t i = 0; i < metadatas_.size(); i++) {
    Metadata *metadata = &metadatas_[i];
    mem::Free(metadata->data);
    DEBUG_SAME_INT(metadatas_[i].lock_count, 0);
    DEBUG_POISON_PTR(metadata->data);
  }
  delete schema_;
}

void SmallCache::Clear(mode_t new_mode) {
  for (index_t i = 0; i < metadatas_.size(); i++) {
    Metadata *metadata = &metadatas_[i];
    DEBUG_SAME_INT(metadata->lock_count, 0);
    mem::Free(metadata->data);
    metadata->data = NULL;
    metadata->lock_count = 0;
  }

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
    schema_->BlockThaw(blockid, 0, n_block_bytes_, data);
  } else {
    // In a real cache implementation, we'd only do this if we knew the block
    // was brand-new.
    schema_->BlockInitFrozen(blockid, 0, n_block_bytes_, data);
    schema_->BlockThaw(blockid, 0, n_block_bytes_, data);
  }
  metadata->data = data;
}

void SmallCache::Writeback_(blockid_t blockid, offset_t begin, offset_t end) {
  if (begin != end) {
    Metadata *metadata = &metadatas_[blockid];
    char *data = metadata->data;

    DEBUG_BOUNDS(begin, end + 1);

    if (data) {
      size_t n_bytes = end - begin;
      char *buf = data + begin;

      //if (unlikely(metadata->lock_count)) {
      //  FATAL("Cannot flush a range that is currently in use.");
      //}

      if (likely(blockid != HEADER_BLOCKID)) {
        schema_->BlockRefreeze(blockid, 0, n_bytes, buf, buf);
      }
      inner_->Write(blockid, begin, end, buf);
      schema_->BlockThaw(blockid, 0, n_bytes, buf, buf);
    }
  }
}

void SmallCache::Flush(blockid_t begin_block, offset_t begin_offset,
    blockid_t last_block, offset_t end_offset) {
  DEBUG_ASSERT(BlockDevice::can_write(mode_));

  if (BlockDevice::must_write(mode_)) {
    if (unlikely(last_block >= n_blocks_)) {
      n_blocks_ = last_block + 1;
      metadatas_.Resize(n_blocks_);
    }
    if (begin_block == last_block) {
      Writeback_(begin_block, begin_offset, end_offset);
    } else {
      Writeback_(begin_block, begin_offset, n_block_bytes_);
      for (blockid_t i = begin_block + 1; i < last_block; i++) {
        Writeback_(i, 0, n_block_bytes_);
      }
      Writeback_(last_block, 0, end_offset);
    }
  }
}

void SmallCache::Flush() {
  DEBUG_ASSERT(BlockDevice::can_write(mode_));

  if (BlockDevice::must_write(mode_)) {
    for (blockid_t i = 0; i < n_blocks_; i++) {
      Writeback_(i, 0, n_block_bytes_);
    }
  }
}

void SmallCache::Read(blockid_t blockid,
    offset_t begin, offset_t end, char *buf) {
  const char *src = StartRead(blockid);
  size_t n_bytes = end - begin;

  // TODO: consider read-through
  schema_->BlockRefreeze(blockid, begin, n_bytes, src + begin, buf);

  StopRead(blockid);
}

void SmallCache::Write(blockid_t blockid,
    offset_t begin, offset_t end, const char *buf) {
  // TODO: Consider write-through on straddles (avoiding the read and
  // cache pollution)
  // TODO: Consider skipping the read when entire block is written (though
  // still may consider keeping block in cache)
  // TODO: This behavior may be necessary due to the specious behavior of
  // Flush et al.
  char *dest = StartWrite(blockid);
  size_t n_bytes = end - begin;

  mem::CopyBytes(dest + begin, buf, n_bytes);
  schema_->BlockThaw(blockid, begin, n_bytes, dest + begin);

  StopWrite(blockid);
}

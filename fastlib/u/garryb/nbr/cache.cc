#include "cache.h"

void SmallCache::Init(
    BlockDevice *inner_in, Schema *handler_in, mode_t mode_in) {
  BlockDeviceWrapper::Init(inner_in); // sets inner_, n_block_bytes_, etc

  metadatas_.Init(n_blocks_);
  handler_ = handler_in;
  mode_ = mode_in;

  ArrayList<char> header_data;
  header_data.Init(inner_->n_block_bytes());

  if (mode_ == CREATE || mode_ == TEMP) {
    char *data = StartWrite(HEADER_BLOCKID);
    handler_->WriteHeader(n_block_bytes_, data);
    StopWrite(HEADER_BLOCKID);
  } else if (mode_ == READ || mode_ == MODIFY) {
    char *data = StartRead(HEADER_BLOCKID);
    handler_->InitFromHeader(n_block_bytes_, data);
    StopRead(HEADER_BLOCKID);
  } else {
    FATAL("Unknown mode");
  }
}

SmallCache::~SmallCache() {
  for (index_t i = 0; i < metadatas_.size(); i++) {
    Metadata *metadata = &metadatas_[i];
    mem::Free(metadata->data);
    DEBUG_SAME_INT(metadatas_[i].lock_count, 0);
    DEBUG_POISON_PTR(metadata->data);
  }
  delete handler_;
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
  if (mode_ == BlockDevice::READ || mode_ == BlockDevice::MODIFY) {
    inner_->Read(blockid, data);
    if (likely(blockid != HEADER_BLOCKID)) {
      handler_->BlockThaw(n_block_bytes_, data);
    }
  } else {
    // In a real cache implementation, we'd only do this if we knew the block
    // was brand-new.
    if (likely(blockid != HEADER_BLOCKID)) {
      handler_->BlockInitFrozen(n_block_bytes_, data);
      handler_->BlockThaw(n_block_bytes_, data);
    }
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
        handler_->BlockRefreeze(n_bytes, buf, buf);
      }
      inner_->Write(blockid, begin, end, buf);
      handler_->BlockThaw(n_bytes, buf);
    }
  }
}

void SmallCache::Flush(blockid_t begin_block, offset_t begin_offset,
    blockid_t last_block, offset_t end_offset) {
  DEBUG_ASSERT(mode_ != BlockDevice::READ);

  if (unlikely(mode_ == BlockDevice::MODIFY || mode_ == BlockDevice::CREATE)) {
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

void SmallCache::Read(blockid_t blockid,
    offset_t begin, offset_t end, char *buf) {
  const char *src_buffer = StartRead(blockid) + begin;
  size_t n_bytes = end - begin;
  mem::CopyBytes(buf, src_buffer, n_bytes);
  // TODO: Allow handler to assert that accesses are on block boundaries
  if (blockid != HEADER_BLOCKID) {
    handler_->BlockRefreeze(n_bytes, src_buffer, buf);
  }
  StopRead(blockid);
}

void SmallCache::Write(blockid_t blockid,
    offset_t begin, offset_t end, const char *buf) {
  DEBUG_ASSERT_MSG(blockid != HEADER_BLOCKID,
      "The header block is protected and non-writable, "
      "because it contains important metadata.");
  
  char *dest_buffer = StartWrite(blockid) + begin;
  size_t n_bytes = end - begin;
  mem::CopyBytes(dest_buffer, buf, n_bytes);
  handler_->BlockThaw(n_bytes, dest_buffer);
  StopWrite(blockid);
}

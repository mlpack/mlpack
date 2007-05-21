#include "cache.h"

void SmallCache::Init(BlockDevice *inner_in, BlockActionHandler *handler_in,
    mode_t mode_in) {
  BlockDeviceWrapper::Init(inner_in);
  metadatas_.Init(inner_in->n_blocks());
  handler_ = handler_in;
  mode_ = mode_in;
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
  metadata->is_dirty = 1;
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
  StopRead(blockid);
}

void SmallCache::PerformCacheMiss_(blockid_t blockid) {
  char *data;
  Metadata *metadata = &metadatas_[blockid];

  data = mem::Alloc<char>(n_block_bytes());
  if (mode_ == BlockDevice::READ || mode_ == BlockDevice::MODIFY) {
    inner_->Read(blockid, data);
    handler_->BlockThaw(n_block_bytes_, data);
  } else {
    handler_->BlockInitFrozen(n_block_bytes_, data);
    handler_->BlockThaw(n_block_bytes_, data);
  }
  metadata->data = data;
}

void SmallCache::Writeback_(blockid_t blockid, offset_t begin, offset_t end) {
  if (begin != end) {
    Metadata *metadata = &metadatas_[blockid];
    char *data = metadata->data;
    
    if (data) {
      size_t n_bytes = end - begin;
      char *buf = data + begin;
      
      if (unlikely(metadata->lock_count)) {
        FATAL("Cannot flush a range that is currently in use.");
      }
      
      handler_->BlockRefreeze(n_bytes, buf, buf);
      inner_->Write(blockid, begin, end, buf);
      handler_->BlockThaw(n_bytes, buf);
    }
  }
}

void SmallCache::Flush(blockid_t begin_block, offset_t begin_offset,
    blockid_t last_block, offset_t end_offset) {
  DEBUG_ASSERT(mode_ != BlockDevice::READ);

  if (unlikely(mode_ == BlockDevice::MODIFY || mode_ == BlockDevice::CREATE)) {
    if (begin_block == last_block) {
      Writeback_(begin_block, begin_offset, end_offset);
    } else {
      Writeback_(begin_block, begin_offset, n_block_bytes_ - begin_offset);
      for (blockid_t i = begin_block + 1; i < last_block - 1; i++) {
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
  handler_->BlockRefreeze(n_bytes, src_buffer, buf);
  StopRead(blockid);
}

void SmallCache::Write(blockid_t blockid,
    offset_t begin, offset_t end, const char *buf) {
  char *dest_buffer = StartWrite(blockid) + begin;
  size_t n_bytes = end - begin;
  mem::CopyBytes(dest_buffer, buf, n_bytes);
  handler_->BlockThaw(n_bytes, dest_buffer);
  StopWrite(blockid);
}

void SmallCache::Close() {
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

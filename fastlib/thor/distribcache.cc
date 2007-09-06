/**
 * @file distribcache.cc
 *
 * Implementation of the distributed cache.
 */

#include "distribcache.h"

#include <stdio.h>

//-------------------------------------------------------------------------
//-- THE DISTRIBUTED CACHE ------------------------------------------------
//-------------------------------------------------------------------------

#warning perform randomized syncing to avoid contention

void DistributedCache::InitMaster(int channel_num_in,
    offset_t n_block_bytes_in,
    size_t total_ram,
    BlockHandler *handler_in) {
  InitCommon_(channel_num_in);
  handler_ = handler_in;
  n_blocks_ = 0;
  n_block_bytes_ = n_block_bytes_in;
  InitFile_(NULL);
  InitCache_(total_ram);
  InitChannel_();
}

void DistributedCache::InitWorker(
    int channel_num_in, size_t total_ram, BlockHandler *handler_in) {
  InitCommon_(channel_num_in);
  // connect to master and figure out specs
  handler_ = handler_in;
  DoConfigRequest_();
  InitFile_(NULL);
  InitCache_(total_ram);
  InitChannel_();
}

DistributedCache::~DistributedCache() {
  delete handler_;
  delete overflow_device_;
  rpc::Unregister(channel_num_);
  
#ifdef DEBUG
  for (index_t i = 0; i < blocks_.size(); i++) {
    DEBUG_SAME_INT(blocks_[i].locks, 0);
  }
#endif
}

void DistributedCache::InitFile_(const char *filename) {
  DiskBlockDevice *db = new DiskBlockDevice();
  db->Init(filename, BlockDevice::M_TEMP, n_block_bytes_);
  overflow_device_ = db;
}

void DistributedCache::InitChannel_() {
  channel_.Init(this);
  rpc::Register(channel_num_, &channel_);
}

void DistributedCache::DoConfigRequest_() {
  BasicTransaction transaction;
  transaction.Init(channel_num_);
  // WALDO -- This used to be sent to MASTER_RANK
  DEBUG_ASSERT(!rpc::is_root());
  Message *message = transaction.CreateMessage(rpc::parent(), sizeof(Request));
  Request *request = message->data_as<Request>();

  request->type = Request::CONFIG;
  request->field1 = 0;
  request->field2 = 0;
  request->field3 = 0;
  transaction.Send(message);
  transaction.WaitDone();
  ConfigResponse *response = ot::PointerThaw<ConfigResponse>(
      transaction.response()->data());

  n_blocks_ = 0;
  n_block_bytes_ = response->n_block_bytes;
  handler_->Deserialize(response->block_handler_data);
}

void DistributedCache::InitCommon_(int channel_num_in) {
  channel_num_ = channel_num_in;
  syncing_ = false;
  disk_stats_.Init();
  net_stats_.Init();
  world_disk_stats_.Init();
  world_net_stats_.Init();
  n_locks_ = 0;
  world_n_locks_ = 0;
  n_fifo_locks_ = 0;
  world_n_fifo_locks_ = 0;

  blocks_.Init();
  handler_ = NULL;

  overflow_free_ = -1;
  overflow_next_.Init();
  overflow_next_.default_value() = BIG_BAD_NUMBER;
  overflow_device_ = NULL;

  write_ranges_.Init();

  my_rank_ = rpc::rank();
}

void DistributedCache::InitCache_(size_t total_ram) {
  // give minimum number of cache sets
  n_sets_ = (total_ram) / (ASSOC*n_block_bytes_);
  if (rpc::n_peers() != 1 && n_sets_ % rpc::n_peers() == 0) {
    n_sets_--;
  }
  if (n_sets_ == 0) {
    NONFATAL("%lu bytes is too small a cache size -- upping size to %lu!",
        (unsigned long)total_ram, (unsigned long)(ASSOC*n_block_bytes_));
    n_sets_ = 1;
  } else {
    DEBUG_ASSERT(n_sets_ * ASSOC * n_block_bytes_ <= total_ram);
  }
  slots_.Init(n_sets_ << LOG_ASSOC);
}

char *DistributedCache::AllocBlock_() {
  /*char *item = block_freelist_;
  if (unlikely(item == NULL)) {
    int slab_items = 128;
    item = mem::Alloc<char>(slab_items * n_block_bytes_);

    *reinterpret_cast<char**>(item) = block_freelist_;
    --slab_items;

    do {
      char *prev = item;
      item += n_block_bytes_;
      *reinterpret_cast<char**>(item) = prev;
    } while (--slab_items);
  }
  block_freelist_ = *reinterpret_cast<char**>(item);
  return item;*/
  return mem::Alloc<char>(n_block_bytes_);
}

void DistributedCache::FreeBlock_(char *item) {
  //*reinterpret_cast<char**>(item) = block_freelist_;
  //block_freelist_ = item;
  mem::Free(item);
}

void DistributedCache::HandleSyncInfo_(const SyncInfo& info) {
  mutex_.Lock();
  HandleStatusInformation_(info.statuses);
  world_disk_stats_ = info.disk_stats;
  world_net_stats_ = info.net_stats;
  world_n_locks_ = info.n_locks;
  world_n_fifo_locks_ = info.n_fifo_locks;
  mutex_.Unlock();
}

void DistributedCache::HandleStatusInformation_(
    const ArrayList<BlockStatus>& statuses) {
  // This method is only called after a sync.
  // However, it is possible that some other machines might have started
  // writing stuff, so we'll have to take this information with a grain of
  // salt.
  DEBUG_ASSERT(statuses.size() >= n_blocks_);

  if (n_blocks_ != statuses.size()) {
    n_blocks_ = statuses.size();
    blocks_.Resize(n_blocks_);
  }

  for (index_t i = 0; i < n_blocks_; i++) {
    BlockMetadata *block = &blocks_[i];
    const BlockStatus *status = &statuses[i];

    if (unlikely(block->is_owner())) {
      // I know I own the block.  Either I already owned the block, or during
      // the duration of the barrier someone else started writing blocks to
      // me and made me the owner of the block (perfectly valid!).
      // Don't do anything, because *I* always have the correct information
      // about my own blocks, and it's okay if I'm getting invalid
      // information.
    } else {
      // A block that I don't own can't be dirty, because locally I'm still
      // performing the sync barrier and couldn't have written to the block
      // myself, and if some other machine had written it, I'd actually
      // be the owner.
      DEBUG_ASSERT_MSG(status->owner != my_rank_,
          "Received ownership unexpectedly");
      DEBUG_ASSERT_MSG(status->owner >= 0,
          "%d It looks like block %"LI"d is owned by %d of %"LI"d (i'm %d)\n",
          int(n_block_bytes_), i, status->owner, blocks_.size(), rpc::rank());
      DEBUG_ASSERT_MSG(!block->is_dirty(),
          "Remote blocks shouldn't be dirty during a sync.");
      block->value = ~status->owner;
      block->status = status->is_new ? NOT_DIRTY_NEW : NOT_DIRTY_OLD;
    }
  }
}

void DistributedCache::ComputeStatusInformation_(
    ArrayList<BlockStatus> *statuses) const {
  mutex_.Lock();
  DEBUG_ASSERT(n_blocks_ == blocks_.size());
  statuses->Init(n_blocks_);
  for (index_t i = 0; i < statuses->size(); i++) {
    BlockStatus *status = &(*statuses)[i];
    const BlockMetadata *block = &blocks_[i];
    if (block->is_owner()) {
      status->owner = my_rank_;
      status->is_new = block->is_new();
    } else {
      status->owner = -1;
      status->is_new = false;
    }
  }
  mutex_.Unlock();
}

void DistributedCache::BestEffortWriteback(double portion) {
  mutex_.Lock();
  Slot *slot = slots_.begin();
  index_t i = slots_.size();
  int start_col = math::RoundInt(ASSOC * (1 - portion));
  BlockMetadata *blocks = blocks_.begin();

  // Might want to software-pipeline this loop, because of the really nasty
  // indirect load going on.
  do {
    i -= ASSOC;
    for (int j = start_col; j < ASSOC; j++) {
      blockid_t blockid = slot[j].blockid;
      if (blockid >= 0) {
        BlockMetadata *block = &blocks[blockid];
        DEBUG_ASSERT_MSG(!block->is_busy(), "Why is a busy block in LRU?");
        DEBUG_ASSERT(!block->is_reading);
        if (block->is_dirty() && !block->is_owner()) {
          WritebackDirtyRemote_(blockid, block->data);
        }
      }
    }
    slot += ASSOC;
  } while (i != 0);
  mutex_.Unlock();
}

void DistributedCache::StartSync() {
  // We'll assume everything we have locally is no longer valid.
  mutex_.Lock();
  Slot *slot = slots_.begin();
  index_t i = slots_.size();
  BlockMetadata *blocks = blocks_.begin();
  size_t unflushed_bytes = 0;

  DEBUG_ASSERT_MSG(!syncing_, "Called StartSync twice before WaitSync!");

  // Might want to software-pipeline this loop, because of the really nasty
  // indirect load going on.
  do {
    i--;
    blockid_t blockid = slot->blockid;
    if (blockid >= 0) {
      BlockMetadata *block = &blocks[blockid];
      if (!block->is_owner()) {
        slot->blockid = -1;
        DEBUG_ASSERT_MSG(block->locks == 0, "Why is a locked block in LRU?");
        DEBUG_ASSERT(!block->is_reading);
        unflushed_bytes += n_block_bytes_;
        // Flush every 4 megabytes to avoid eating RAM with buffers.
        if (unflushed_bytes > 4*MEGABYTE) {
          rpc::WriteFlush();
        }
        Purge_(blockid);
        DEBUG_ASSERT_MSG(!block->is_dirty(),
            "We purged a block and it's still marked as dirty?");
      }
    }
    slot++;
  } while (i != 0);

#ifdef DEBUG
  for (index_t i = 0; i < n_blocks_; i++) {
    BlockMetadata *block = &blocks[i];
    if (block->is_busy() || block->is_dirty()) {
      DEBUG_ASSERT(block->is_in_core());
      DEBUG_ASSERT_MSG(block->is_owner(),
          "During a sync point, all busy blocks can only be local blocks.");
    }
  }
#endif

  // TODO: Make absolutely certain nobody is currently accessesing the cache
  write_ranges_.Reset();
  syncing_ = true;
  mutex_.Unlock();

  // make sure none of these writes are still in flight
  rpc::WriteFlush();

  channel_.StartSyncFlushDone();
}

void DistributedCache::WaitSync(datanode *node) {
  channel_.WaitSync();

  mutex_.Lock();
  syncing_ = false;
  if (node) {
    world_disk_stats().Report(n_block_bytes_, n_blocks_, 
        fx_submodule(node, NULL, "world_disk_stats"));
    if (rpc::n_peers() > 1) {
      // net stats are only interesting if there's at least two machines
      world_net_stats().Report(n_block_bytes_, n_blocks_, 
          fx_submodule(node, NULL, "world_net_stats"));
    }
    #ifdef DEBUG
    fx_format_result(node, "world_n_locks", "%"L64"d", world_n_locks_);
    fx_format_result(node, "world_lock_ratio", "%f",
        1.0 * world_n_locks_ / n_blocks_);
    fx_format_result(node, "world_n_fifo_locks", "%"L64"d", world_n_fifo_locks_);
    fx_format_result(node, "world_fifo_miss_ratio", "%f",
        1.0 * world_n_locks_ / world_n_fifo_locks_);
    #endif
  }
  n_locks_ = 0;
  n_fifo_locks_ = 0;
  disk_stats_.Reset();
  net_stats_.Reset();
  mutex_.Unlock();
}

void DistributedCache::ResetElements() {
  mutex_.Lock();
  for (index_t blockid = 0; blockid < n_blocks_; blockid++) {
    BlockMetadata *block = &blocks_[blockid];

    DEBUG_ASSERT_MSG(!block->is_busy(),
        "Cannot reset elements if some blocks are busy.");
    DEBUG_ASSERT(!block->is_reading);

    if (block->is_in_core()) {
      FreeBlock_(block->data);
      block->data = NULL;
    }
    if (block->is_owner()) {
      RecycleLocalBlock_(block->local_blockid());
    }
    
    
    block->status = NOT_DIRTY_NEW;
  }
  for (index_t i = slots_.size(); i--;) {
    slots_[i].blockid = -1;
  }
  write_ranges_.Reset();
  mutex_.Unlock();
}

//----

void DistributedCache::Read(blockid_t blockid,
    offset_t begin, offset_t end, char *buf) {
  mutex_.Lock();
  BlockMetadata *block = &blocks_[blockid];
  if (unlikely(block->locks == 0)) {
    DecacheBlock_(blockid);
    block->locks = 0;
  }
  offset_t n_bytes = end - begin;
  mem::CopyBytes(buf, block->data + begin, n_bytes);
  handler_->BlockFreeze(blockid, begin, n_bytes, block->data + begin, buf);
  if (unlikely(block->locks == 0)) {
    EncacheBlock_(blockid);
  }
  mutex_.Unlock();
}

void DistributedCache::RemoteRead(blockid_t blockid,
    offset_t begin, offset_t end, char *buf) {
#ifdef DEBUG
  mutex_.Lock();
  DEBUG_ASSERT_MSG(blocks_[blockid].is_owner(),
      "Remote reads must be sent to the block's owner -- it looks like the "
      "block mapping has gotten out of sync.  Remember to sync all machines "
      "after a number of block mapping changes.");
  mutex_.Unlock();
#endif
  Read(blockid, begin, end, buf);
}

void DistributedCache::Write(blockid_t blockid,
    offset_t begin, offset_t end, const char *buf) {
  // i have to be the owner of the block
  char *dest = StartWrite(blockid, false) + begin;
  size_t n_bytes = end - begin;

  mem::CopyBytes(dest, buf, n_bytes);
  handler_->BlockThaw(blockid, begin, n_bytes, dest);

  StopWrite(blockid);
}

void DistributedCache::RemoteWrite(blockid_t blockid,
    offset_t begin, offset_t end, const char *buf) {
  mutex_.Lock();
  if (unlikely(blockid >= n_blocks_)) {
    n_blocks_ = blockid + 1;
    // the default constructor for BlockMetadata should mark the block as new
    blocks_.Resize(n_blocks_);
  }
  BlockMetadata *block = &blocks_[blockid];
  if (!block->is_owner()) {
    // when we receive a remote write, we are always the owner
    DEBUG_ASSERT_MSG(!block->is_reading,
        "One machine is reading a block, but simultaneously received ownership.\n"
        "Please sync between ownership changes and further reads!");
    block->value = SELF_OWNER_UNALLOCATED; // mark as owner
    block->status = NOT_DIRTY_NEW;
    DEBUG_ASSERT(block->is_owner());
  }
  mutex_.Unlock();
  Write(blockid, begin, end, buf);
}

BlockDevice::blockid_t DistributedCache::AllocBlocks(
    blockid_t n_blocks_to_alloc, int owner) {
  index_t blockid = RemoteAllocBlocks(n_blocks_to_alloc, owner, my_rank_);
  if (owner != my_rank_) {
    // Tell the owner that I've allocated a block in their name.
    DoOwnerRequest_(owner, owner, blockid, blockid + n_blocks_to_alloc);
  }
  return blockid;
}

BlockDevice::blockid_t DistributedCache::RemoteAllocBlocks(
    blockid_t n_blocks_to_alloc, int owner, int sender) {
  blockid_t blockid;

  if (likely(my_rank_ == MASTER_RANK)) {
    // Append some blocks to the end
    mutex_.Lock();
    blockid = n_blocks_;
  } else {
    blockid = DoAllocRequest_(n_blocks_to_alloc, owner);
    mutex_.Lock();
  }

  n_blocks_ = blockid + n_blocks_to_alloc;
  blocks_.GrowTo(n_blocks_);
  // these blocks are marked as NOT_DIRTY_NEW
  
  MarkOwner_(owner, blockid, n_blocks_);
  mutex_.Unlock();

  return blockid;
}

void DistributedCache::MarkOwner_(int owner,
    blockid_t begin, blockid_t end) {
  int32 value = (owner == my_rank_) ? SELF_OWNER_UNALLOCATED : (~owner);

  for (blockid_t i = begin; i < end; i++) {
    if (blocks_[i].is_owner() && value < 0) {
      blocks_[i].status = NOT_DIRTY_OLD;
      RecycleLocalBlock_(blocks_[i].local_blockid());
    }
    blocks_[i].value = value;
  }
}

BlockDevice::blockid_t DistributedCache::DoAllocRequest_(
    blockid_t n_blocks_to_alloc, int owner) {
  BasicTransaction transaction;
  transaction.Init(channel_num_);
  Message *message = transaction.CreateMessage(MASTER_RANK, sizeof(Request));
  Request *request = message->data_as<Request>();
  request->type = Request::ALLOC;
  request->field1 = n_blocks_to_alloc;
  request->field2 = owner;
  request->field3 = 0;
  transaction.Send(message);
  transaction.WaitDone();
  blockid_t retval = *transaction.response()->data_as<blockid_t>();
  return retval;
}

void DistributedCache::DoOwnerRequest_(int dest, int new_owner,
    blockid_t blockid, blockid_t end_block) {
  BasicTransaction transaction;
  transaction.Init(channel_num_);
  Message *message = transaction.CreateMessage(dest, sizeof(Request));
  Request *request = message->data_as<Request>();
  request->type = Request::OWNER;
  request->field1 = blockid;
  request->field2 = end_block;
  request->field3 = new_owner;
  transaction.Send(message);
  transaction.Done();
}

void DistributedCache::HandleRemoteOwner_(blockid_t block, blockid_t end,
    int new_owner) {
  mutex_.Lock();
  n_blocks_ = max(n_blocks_, end);
  blocks_.Resize(n_blocks_);
  MarkOwner_(new_owner, block, end);
  mutex_.Unlock();
}

void DistributedCache::GiveOwnership(blockid_t blockid, int new_owner) {
  // mark whole block as dirty and change its owner.
  mutex_.Lock();
  BlockMetadata *block = &blocks_[blockid];
  if (block->owner(this) != new_owner) {
    if (unlikely(block->locks == 0)) {
      DecacheBlock_(blockid);
      block->locks = 0;
    }
    if (!block->is_owner()) {
      DoOwnerRequest_(block->owner(), new_owner, blockid, blockid + 1);
    } else {
      RecycleLocalBlock_(block->local_blockid());
    }
    block->value = (new_owner == my_rank_) ? SELF_OWNER_UNALLOCATED : (~new_owner);
    block->status = FULLY_DIRTY;
    if (unlikely(block->locks == 0)) {
      EncacheBlock_(blockid);
    }
  }
  mutex_.Unlock();
}

void DistributedCache::RecycleLocalBlock_(blockid_t local_blockid) {
  // this block has a location on disk -- since it's not ours anymore,
  // recycle its allocated disk space.
  if (local_blockid != SELF_OWNER_UNALLOCATED) {
    overflow_next_[local_blockid] = overflow_free_;
    overflow_free_ = local_blockid;
  }
}

//----

char *DistributedCache::StartWrite(blockid_t blockid, bool is_partial) {
  mutex_.Lock();
  DEBUG_ONLY(n_locks_++);
  BlockMetadata *block = &blocks_[blockid];
  if (likely(block->locks)) {
    block->locks++;
  } else {
    DecacheBlock_(blockid);
  }
  if (is_partial) {
    block->status &= PARTIALLY_DIRTY;
  } else {
    block->status = FULLY_DIRTY;
  }
  mutex_.Unlock();
  return block->data;
}

char *DistributedCache::StartRead(blockid_t blockid) {
  mutex_.Lock();
  DEBUG_ONLY(n_locks_++);
  BlockMetadata *block = &blocks_[blockid];
  if (likely(block->locks)) {
    block->locks++;
  } else {
    DecacheBlock_(blockid);
  }
  mutex_.Unlock();
  return block->data;
}

void DistributedCache::DecacheBlock_(blockid_t blockid) {
  BlockMetadata *block = &blocks_[blockid];
  index_t slot = (unsigned(blockid) % unsigned(n_sets_)) << LOG_ASSOC;
  Slot *base_slot = &slots_[slot];

  DEBUG_ASSERT(!block->is_busy());

  if (likely(block->is_in_core())) {
    DEBUG_ASSERT(!block->is_reading);
    // It's in core, but its lock count was zero, so that means it's
    // definitely definitely in cache and in this line.
    for (int i = 0;; i++) {
      DEBUG_ASSERT_MSG(i != ASSOC, "Couldn't find %d in cache", blockid);
      if (unlikely(base_slot[i].blockid == blockid)) {
        base_slot[i].blockid = -1;
        break;
      }
    }
    block->locks = 1;
  } else {
    HandleMiss_(blockid);
  }
}

void DistributedCache::HandleMiss_(blockid_t blockid) {
  BlockMetadata *block = &blocks_[blockid];

  DEBUG_ASSERT(block->data == NULL);
  DEBUG_ASSERT(block->locks == 0);

  if (block->is_reading) {
    // Increase is_reading so that the busy thread will wake up the I/O
    // condition.
    block->is_reading = WAITING;

    // The block is currently being read.
    // Note instead of storing a whole mutex for each block, we store
    // is_reading and have a global I/O condition, which together
    // simulate a mutex.
    while (block->is_reading != NOT_READING) {
      io_cond_[blockid % IO_COND_MODULO].Wait(&mutex_);
    }

    // We're starting from scratch, recursively calling DecacheBlock.
    // In the time that we received the signal, practically anything could
    // have happened to the block -- it might even be gone completely from
    // cache and gone back to the remote host (though very unlikely).
    if (likely(block->locks)) {
      block->locks++;
    } else {
      return DecacheBlock_(blockid); // tail call
    }
  } else {
    // We're exclusive now -- nobody else is reading the block.

    if (block->is_new()) {
      block->data = AllocBlock_();
      DEBUG_ASSERT_MSG(block->status == NOT_DIRTY_NEW,
          "Block should be NOT_DIRTY_NEW, because that's what is_new() means");
      handler_->BlockInitFrozen(blockid, 0, n_block_bytes_, block->data);
    } else {
      DEBUG_ASSERT(block->status == NOT_DIRTY_OLD);
      HandleRealMiss_(blockid);
    }

    handler_->BlockThaw(blockid, 0, n_block_bytes_, block->data);
    DEBUG_ASSERT(block->locks == 0);
    block->locks = 1;
  }
}

void DistributedCache::HandleRealMiss_(blockid_t blockid) {
  BlockMetadata *block = &blocks_[blockid];
  int value = block->value;

  DEBUG_ASSERT(!block->is_reading);
  block->is_reading = READING;
  mutex_.Unlock();

  char *data = AllocBlock_();
  if (value >= 0) {
    blockid_t local_blockid = value;
    // read the block from disk
    disk_stats_.RecordRead(n_block_bytes_);
    overflow_device_->Read(local_blockid, 0, n_block_bytes_, data);
  } else {
    int owner = ~value;
    // read the block from a remote machine
    net_stats_.RecordRead(n_block_bytes_);
    DoReadRequest_(owner, blockid, 0, n_block_bytes_, data);
  }

  mutex_.Lock();
  int is_reading = block->is_reading;
  DEBUG_ASSERT(block->data == NULL);
  block->is_reading = NOT_READING;
  block->data = data;
  if (is_reading == WAITING) {
    // I wasn't the only reader, broadcast the block's status
    io_cond_[blockid % IO_COND_MODULO].Broadcast();
  }
}

void DistributedCache::DoReadRequest_(int peer, blockid_t blockid,
    offset_t begin, offset_t end, char *buffer) {
  BasicTransaction transaction;
  transaction.Init(channel_num_);
  Message *message = transaction.CreateMessage(peer, sizeof(Request));
  Request *request = message->data_as<Request>();
  request->type = Request::READ;
  request->field1 = blockid;
  request->field2 = begin;
  request->field3 = end;
  transaction.Send(message);
  transaction.WaitDone();
  mem::CopyBytes(buffer, transaction.response()->data(), end - begin);
}

void DistributedCache::StopRead(blockid_t blockid) {
  mutex_.Lock();
  BlockMetadata *block = &blocks_[blockid];
  if (unlikely(--block->locks == 0)) {
    EncacheBlock_(blockid);
  }
  mutex_.Unlock();
}

void DistributedCache::StopWrite(blockid_t blockid) {
  mutex_.Lock();
  BlockMetadata *block = &blocks_[blockid];
  if (unlikely(--block->locks == 0)) {
    EncacheBlock_(blockid);
  }
  mutex_.Unlock();
}

void DistributedCache::EncacheBlock_(blockid_t blockid) {
  index_t slot = (unsigned(blockid) % unsigned(n_sets_)) << LOG_ASSOC;
  Slot *base_slot = &slots_[slot];
  int i;

  // Find first unused slot and move to front.
  i = 0;

  while (base_slot[i].blockid >= 0) {
    // Make sure a block isn't in cache twice
    DEBUG_ASSERT_MSG(base_slot[i].blockid != blockid,
        "Block re-cached: block %d, cache block size %d",
        blockid, n_block_bytes_);
    if (unlikely(i == ASSOC-1)) {
      for (;;) {
        if (!blocks_[base_slot[i].blockid].is_owner()) {
          break;
        }
        i--;
        if (i < REMOTE_ALLOWANCE) {
          // no renote block to purge, take the worst local one
          i = ASSOC-1;
          break;
        }
      }
      Purge_(base_slot[i].blockid);
      DEBUG_ONLY(base_slot[i].blockid = -1);
      break;
    }
    i++;
  }

  DEBUG_ASSERT(base_slot[i].blockid == -1);
  for (; i != 0; i--) {
    base_slot[i] = base_slot[i-1];
  }

  base_slot[0].blockid = blockid;
}

void DistributedCache::Purge_(blockid_t blockid) {
  BlockMetadata *block = &blocks_[blockid];
  char *data = block->data;

  DEBUG_ASSERT(block->is_in_core());
  DEBUG_ASSERT_MSG(!block->is_busy(),
      "Trying to evict a busy block (non-zero lock count %d, block %d, value %d)",
      int(block->locks), int(blockid), int(block->value));

  block->data = NULL;

  if (block->is_dirty()) {
    if (block->is_owner()) {
      WritebackDirtyLocalFreeze_(blockid, data);
    } else {
      WritebackDirtyRemote_(blockid, data);
    }
  }

  FreeBlock_(data);
}

void DistributedCache::WritebackDirtyLocalFreeze_(
    blockid_t blockid, char *data) {
  BlockMetadata *block = &blocks_[blockid];

  DEBUG_ASSERT(block->is_owner());

  blockid_t local_blockid = block->local_blockid();

  if (local_blockid == SELF_OWNER_UNALLOCATED) {
    local_blockid = overflow_free_;
    if (local_blockid < 0) {
      local_blockid = overflow_device_->AllocBlocks(1);
    } else {
      // free blocks come from calls to GiveOwnership
      overflow_free_ = overflow_next_.get(local_blockid);
    }
    block->value = local_blockid;
  }

  DEBUG_ASSERT(block->is_dirty());
  //fprintf(stderr, "DISK: writing %d to %d (%d bytes)\n",
  //    blockid, local_blockid, n_block_bytes_);
  disk_stats_.RecordWrite(n_block_bytes_);
  block->status = NOT_DIRTY_OLD;

  handler_->BlockFreeze(blockid, 0, n_block_bytes_, data, data);
  overflow_device_->Write(local_blockid, 0, n_block_bytes_, data);
}

void DistributedCache::WritebackDirtyRemote_(blockid_t blockid, char *data) {
  BlockMetadata *block = &blocks_[blockid];

  DEBUG_ASSERT(!block->is_owner());
  DEBUG_ASSERT(block->is_dirty());

  if (block->status == FULLY_DIRTY) {
    // The entire block is dirty
    DoWriteRequest_(block->owner(), blockid,
        0, n_block_bytes_, data);
  } else {
    DEBUG_ASSERT(block->status == PARTIALLY_DIRTY);
    // Find the intersection between this block and all dirty ranges we
    // know about.
    #ifdef DEBUG
    offset_t bytes_total = 0;
    #endif
    for (index_t i = 0; i < write_ranges_.size(); i++) {
      Position begin = write_ranges_[i].begin;
      Position end = write_ranges_[i].end;
      if (blockid >= begin.block && blockid <= end.block) {
        // We found a partial range that overlaps.  Write it.
        offset_t begin_offset = 0;
        offset_t end_offset = n_block_bytes_;
        if (blockid == begin.block) {
          begin_offset = begin.offset;
        }
        if (blockid == end.block) {
          end_offset = end.offset;
        }
        if (end_offset > begin_offset) {
          DoWriteRequest_(block->owner(), blockid,
              begin_offset, end_offset, data + begin_offset);
        }
        DEBUG_ASSERT(end_offset >= begin_offset);
        DEBUG_ONLY(bytes_total += end_offset - begin_offset);
      }
    }
#ifdef DEBUG
    if (unlikely(bytes_total == 0)) {
      ot::Print(write_ranges_);
    }
#endif
    DEBUG_ASSERT_MSG(bytes_total != 0,
        "%d: A block marked partially-dirty has no overlapping write ranges: block %d, %d blocks total, %d bytes per block.",
        my_rank_, blockid, n_blocks_, n_block_bytes_);
    DEBUG_ASSERT_MSG(bytes_total <= n_block_bytes_,
        "A block marked partially-dirty was written more than once: %d > %d.",
        bytes_total, n_block_bytes_);
  }
  block->status = NOT_DIRTY_OLD;
}

void DistributedCache::DoWriteRequest_(
    int peer, blockid_t blockid, offset_t begin, offset_t end,
    const char *buffer) {
  net_stats_.RecordWrite(end - begin);
  BasicTransaction transaction;
  transaction.Init(channel_num_);
  offset_t n_bytes = end - begin;
  Message *message = transaction.CreateMessage(peer, Request::size(n_bytes));
  Request *request = message->data_as<Request>();
  request->type = Request::WRITE;
  request->field1 = blockid;
  request->field2 = begin;
  request->field3 = end;
  mem::CopyBytes(request->data_as<char>(), buffer, n_bytes);
  handler_->BlockFreeze(blockid, begin, n_bytes, buffer, request->data_as<char>());
  transaction.Send(message);
  transaction.Done();
}

void DistributedCache::AddPartialDirtyRange(
    blockid_t begin_block, offset_t begin_offset,
    blockid_t last_block, offset_t end_offset) {
  mutex_.Lock();
  Position begin;
  Position end;
  begin.block = begin_block;
  begin.offset = begin_offset;
  end.block = last_block;
  end.offset = end_offset;
  write_ranges_.Union(begin, end);
  mutex_.Unlock();
}

//-------------------------------------------------------------------------
//-- PROTOCOL MESSAGES ----------------------------------------------------
//-------------------------------------------------------------------------
//-------------------------------------------------------------------------

void DistributedCache::ResponseTransaction::Init(
    DistributedCache *cache_in) {
  cache_ = cache_in;
  Transaction::Init(cache_->channel_num());
}

void DistributedCache::ResponseTransaction::HandleMessage(
    Message *message) {
  Request *request = reinterpret_cast<Request*>(message->data());

  switch (request->type) {
    case Request::CONFIG: {
      ConfigResponse config_response;
      cache_->handler_->Serialize(&config_response.block_handler_data);
      config_response.n_block_bytes = cache_->n_block_bytes_;
      Message *response = CreateMessage(message->peer(),
          ot::PointerFrozenSize(config_response));
      ot::PointerFreeze(config_response, response->data());
      Send(response);
    }
    break;
    case Request::READ: {
      blockid_t blockid = request->field1;
      offset_t begin = request->field2;
      offset_t end = request->field3;
      Message *response = CreateMessage(message->peer(), end - begin);
      cache_->RemoteRead(blockid, begin, end, response->data());
      Send(response);
    }
    break;
    case Request::WRITE: {
      blockid_t blockid = request->field1;
      offset_t begin = request->field2;
      offset_t end = request->field3;
      cache_->RemoteWrite(blockid, begin, end, request->data_as<char>());
    }
    break;
    case Request::OWNER: {
      blockid_t blockid = request->field1;
      blockid_t end_blockid = request->field2;
      blockid_t owner = request->field3;
      cache_->HandleRemoteOwner_(blockid, end_blockid, owner);
    }
    break;
    case Request::ALLOC: {
      blockid_t blockid = request->field1;
      blockid_t rank = request->field2;
      DEBUG_ASSERT(cache_->my_rank_ == MASTER_RANK);
      Message *response = CreateMessage(message->peer(), sizeof(blockid_t));
      *message->data_as<blockid_t>() =
          cache_->RemoteAllocBlocks(blockid, rank, message->peer());
      Send(response);
    }
    break;
   default:
    FATAL("Unknown DistributedCache message: %d", int(request->type));
  }

  Done();

  delete message;
  delete this;
}

//-------------------------------------------------------------------------

void DistributedCache::SyncInfo::Init(const DistributedCache& cache) {
  disk_stats = cache.disk_stats();
  net_stats = cache.net_stats();
  n_locks = cache.n_locks();
  n_fifo_locks = cache.n_fifo_locks();
  cache.ComputeStatusInformation_(&statuses);
}

void DistributedCache::SyncInfo::MergeWith(const SyncInfo& other) {
  index_t old_size = statuses.size();
  index_t min_size = min(statuses.size(), other.statuses.size());

  for (index_t i = 0; i < min_size; i++) {
    BlockStatus *orig = &statuses[i];
    const BlockStatus *in = &other.statuses[i];
    if (in->owner >= 0) {
      DEBUG_ASSERT(orig->owner < 0);
      *orig = *in;
    }
  }
  if (old_size < other.statuses.size()) {
    statuses.Resize(other.statuses.size());
    mem::Copy(&statuses[old_size], &other.statuses[old_size],
        statuses.size() - old_size);
  }
  disk_stats.Add(other.disk_stats);
  net_stats.Add(other.net_stats);
  n_locks += other.n_locks;
}

//-------------------------------------------------------------------------

void DistributedCache::SyncTransaction::Init(DistributedCache *cache_in) {
  cache_ = cache_in;
  Transaction::Init(cache_->channel_num());
  state_ = CHILDREN_FLUSHING;
  n_ = 0;
}

void DistributedCache::SyncTransaction::HandleMessage(Message *message) {
  mutex_.Lock();
  switch (state_) {
    case CHILDREN_FLUSHING:
      ChildFlushed_();
      break;
    case OTHERS_FLUSHING:
      DEBUG_ASSERT(message->peer() == rpc::parent());
      ParentFlushed_();
      break;
    case CHILDREN_ACCUMULATING:
      AccumulateChild_(message);
      break;
    case OTHERS_ACCUMULATING: {
      DEBUG_ASSERT(message->peer() == rpc::parent());
      char *data = message->data_as<Request>()->data_as<char>();
      SyncInfo *info = ot::PointerThaw<SyncInfo>(data);
      ParentAccumulated_(*info);
    }
    break;
    default:
      FATAL("Unknown state");
  }
  bool is_done = (state_ == DONE);
  mutex_.Unlock();
  delete message;
  if (is_done) {
    delete this;
  }
}

void DistributedCache::SyncTransaction::StartSyncFlushDone() {
  mutex_.Lock();
  ChildFlushed_();
  bool is_done = (state_ == DONE);
  mutex_.Unlock();
  if (is_done) {
    delete this;
  }
}

void DistributedCache::SyncTransaction::ChildFlushed_() {
  n_++;
  // must accumulated n_children + 1: myself!
  if (n_ == rpc::n_children() + 1) {
    if (rpc::is_root()) {
      ParentFlushed_();
    } else {
      state_ = OTHERS_FLUSHING;
      SendBlankSyncMessage_(rpc::parent());
    }
  }
}

void DistributedCache::SyncTransaction::ParentFlushed_() {
  n_ = 0;
  for (index_t i = 0; i < rpc::n_children(); i++) {
    SendBlankSyncMessage_(rpc::child(i));
  }
  // Now we are absolutely certain ALL machines have finished flushing.
  // Thus, our block ownership is accurate.
  sync_info_.Init(*cache_);
  state_ = CHILDREN_ACCUMULATING;
  CheckAccumulation_();
}

void DistributedCache::SyncTransaction::AccumulateChild_(Message *message) {
  SyncInfo *info = ot::PointerThaw<SyncInfo>(
      message->data_as<Request>()->data_as<char>());
  sync_info_.MergeWith(*info);
  n_++;
  CheckAccumulation_();
}

void DistributedCache::SyncTransaction::CheckAccumulation_() {
  // only need to accumulate children, since i myself am a given
  if (n_ == rpc::n_children()) {
    if (rpc::is_root()) {
      ParentAccumulated_(sync_info_);
    } else {
      state_ = OTHERS_ACCUMULATING;
      SendStatusInformation_(rpc::parent(), sync_info_);
    }
  }
}

void DistributedCache::SyncTransaction::ParentAccumulated_(
    const SyncInfo& info) {
  cache_->HandleSyncInfo_(info);
  cache_->channel_.SyncDone();
  for (index_t i = 0; i < rpc::n_children(); i++) {
    SendStatusInformation_(rpc::child(i), info);
  }
  Done();
  state_ = DONE;
}

void DistributedCache::SyncTransaction::SendBlankSyncMessage_(int peer) {
  Message *request_msg = CreateMessage(peer, sizeof(Request));
  Request *request = request_msg->data_as<Request>();
  request->type = Request::SYNC;
  request->field1 = 0;
  request->field2 = 0;
  request->field3 = 0;
  Send(request_msg);
}

void DistributedCache::SyncTransaction::SendStatusInformation_(int peer,
    const SyncInfo& info) {
  // we have two layers of headers here, and then we can freeze the
  // ArrayList into place.
  Message *request_msg = CreateMessage(peer, Request::size(
      ot::PointerFrozenSize(info)));
  Request *request = request_msg->data_as<Request>();
  ot::PointerFreeze(info, request->data_as<char>());
  request->type = Request::SYNC;
  request->field1 = 0;
  request->field2 = 0;
  request->field3 = 0;
  Send(request_msg);
}

//-------------------------------------------------------------------------

void DistributedCache::CacheChannel::Init(DistributedCache *cache_in) {
  cache_ = cache_in;
  sync_transaction_ = NULL;
}

void DistributedCache::CacheChannel::StartSyncFlushDone() {
  SyncTransaction *t = GetSyncTransaction_();;
  t->StartSyncFlushDone();
}

void DistributedCache::CacheChannel::WaitSync() {
  sync_done_.Wait();
}

void DistributedCache::CacheChannel::SyncDone() {
  // this is called by the sync transaction to flag that syncing is done
  mutex_.Lock();
  sync_transaction_ = NULL;
  mutex_.Unlock();
  sync_done_.Done();
}

DistributedCache::SyncTransaction *
DistributedCache::CacheChannel::GetSyncTransaction_() {
  SyncTransaction *t;
  mutex_.Lock();
  if (sync_transaction_ == NULL) {
    sync_transaction_ = new SyncTransaction();
    sync_transaction_->Init(cache_);
  }
  t = sync_transaction_;
  mutex_.Unlock();
  return t;
}

Transaction *DistributedCache::CacheChannel::GetTransaction(
    Message *message) {
  Request *request = reinterpret_cast<Request*>(message->data());
  if (unlikely(request->type == Request::SYNC)) {
    Transaction *t = GetSyncTransaction_();
    return t;
  } else {
    ResponseTransaction *t = new ResponseTransaction();
    t->Init(cache_);
    return t;
  }
}

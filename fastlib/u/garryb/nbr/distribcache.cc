
/*to-do

- Init functions
/- the protocol
  /- nothing is initialized in my version yet
  /- alloc() ?
- dynamic depth
/- the replacement policy
  /- copy from old code (old-cache.c)
  - use static width with tunable depth (that might be based on external
  pressure)
  /- think if it's possible to skip the fifo
    /- no, not possible, no buffer size guarantees
  /- how do we check items out of the n-way cache?
    /- once we start using it, we can just leave a "hole" in the cache set
    which can be filled later, like a victim cache
/- dirty marking needs to be improved
*/

#include "distribcache.h"

#include <stdio.h>

//-------------------------------------------------------------------------
//-- THE DISTRIBUTED CACHE ------------------------------------------------
//-------------------------------------------------------------------------

void DistributedCache::InitMaster(int channel_num_in,
    BlockDevice::offset_t n_block_bytes_in,
    size_t total_ram,
    BlockHandler *handler_in) {
  InitCommon_();
  handler_ = handler_in;
  n_blocks_ = 0;
  n_block_bytes_ = n_block_bytes_in;
  InitFile_(NULL);
  InitCache_(total_ram);
  InitChannel_(channel_num_in);
}

void DistributedCache::InitWorker(
    int channel_num_in, size_t total_ram, BlockHandler *handler_in) {
  InitCommon_();
  // connect to master and figure out specs
  ConfigTransaction ct;
  ConfigResponse *response = ct.Doit(channel_num_in, MASTER_RANK);
  n_blocks_ = 0;
  n_block_bytes_ = response->n_block_bytes;
  handler_ = handler_in;
  handler_->Deserialize(response->block_handler_data);
  InitFile_(NULL);
  InitCache_(total_ram);
  InitChannel_(channel_num_in);
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
  db->Init(filename,
      BlockDevice::M_TEMP, n_block_bytes_);
  overflow_device_ = db;
}

void DistributedCache::InitChannel_(int channel_num_in) {
  channel_num_ = channel_num_in;
  channel_.Init(this);
  rpc::Register(channel_num_, &channel_);
}

void DistributedCache::InitCommon_() {
  disk_stats_.Init();
  net_stats_.Init();
  world_disk_stats_.Init();
  world_net_stats_.Init();

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
  // give enough cache sets, but rounded up
  n_sets_ = (total_ram + ASSOC*n_block_bytes_ - 1) / (ASSOC*n_block_bytes_);
  slots_.Init(n_sets_ << LOG_ASSOC);
}

void DistributedCache::HandleSyncInfo_(const SyncInfo& info) {
  mutex_.Lock();
  HandleStatusInformation_(info.statuses);
  world_disk_stats_ = info.disk_stats;
  world_net_stats_ = info.net_stats;
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
      DEBUG_ASSERT_MSG(!block->is_owner(),
          "Lost ownership unexpectedly");
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
  statuses->Init(n_blocks());
  for (index_t i = 0; i < statuses->size(); i++) {
    BlockStatus *status = &(*statuses)[i];
    const BlockMetadata *block = &blocks_[i];
    if (block->is_owner()) {
      status->owner = my_rank_;
      status->is_new = block->is_new();
    } else {
      status->owner = -1;
      status->is_new = NOT_DIRTY_NEW;
    }
  }
  mutex_.Unlock();
}

void DistributedCache::BestEffortWriteback(double portion) {
  mutex_.Lock();
  Slot *slot = slots_.begin();
  index_t i = slots_.size();
  int start_col = int(nearbyint(ASSOC * portion));
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
        if (unlikely(block->is_dirty()) && unlikely(!block->is_owner())) {
          WritebackDirtyRemote_(blockid);
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

  // Might want to software-pipeline this loop, because of the really nasty
  // indirect load going on.
  do {
    i--;
    blockid_t blockid = slot->blockid;
    if (blockid >= 0) {
      BlockMetadata *block = &blocks[blockid];
      slot++;
      if (unlikely(!block->is_owner())) {
        slot->blockid = -1;
        DEBUG_ASSERT_MSG(block->locks == 0, "Why is a locked block in LRU?");
        Purge_(blockid);
      }
    }
  } while (i != 0);

  // TODO: Make absolutely certain nobody is currently accessesing the cache
  write_ranges_.Reset();
  mutex_.Unlock();

  channel_.StartSyncFlushDone();
}

void DistributedCache::WaitSync() {
  channel_.WaitSync();
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
  mem::Copy(buf, block->data + begin, n_bytes);
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
  // hmm, i have to be the owner of the block
  char *dest = StartWrite(blockid, true);
  size_t n_bytes = end - begin;

  mem::CopyBytes(dest + begin, buf, n_bytes);

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
    block->value = SELF_OWNER_UNALLOCATED; // mark as owner
  }
  mutex_.Unlock();
  Write(blockid, begin, end, buf);
}

BlockDevice::blockid_t DistributedCache::AllocBlocks(
    blockid_t n_blocks_to_alloc, int owner) {
  blockid_t blockid;

  if (likely(my_rank_ == MASTER_RANK)) {
    mutex_.Lock();
    // Append some blocks to the end
    blockid = n_blocks_;
    n_blocks_ += n_blocks_to_alloc;

    // Now, mark the owner of all these blocks.
    BlockMetadata *block = blocks_.AddBack(n_blocks_to_alloc);
    DEBUG_ASSERT(blocks_.size() == n_blocks_);
    int32 value = (owner == my_rank_) ? SELF_OWNER_UNALLOCATED : (~owner);

    for (blockid_t i = 0; i < n_blocks_to_alloc; i++) {
      block[i].value = value;
    }

    mutex_.Unlock();
  } else {
    AllocTransaction t;
    blockid = t.Doit(channel_num_, MASTER_RANK, n_blocks_to_alloc, owner);
    mutex_.Lock();
    n_blocks_ = blockid + n_blocks_to_alloc;
    blocks_.GrowTo(n_blocks_);
    mutex_.Unlock();
  }

  return blockid;
}

void DistributedCache::GiveOwnership(blockid_t my_blockid, int new_owner) {
  if (likely(new_owner != my_rank_)) {
    // mark whole block as dirty and change its owner.
    StartWrite(my_blockid, false);
    mutex_.Lock();
    BlockMetadata *block = &blocks_[my_blockid];
    DEBUG_ASSERT_MSG(block->is_owner(),
        "Can only give ownership if I'm the owner");
    if (block->local_blockid() != SELF_OWNER_UNALLOCATED) {
      // this block has a location on disk -- since it's not ours anymore,
      // recycle its allocated disk space.
      overflow_next_[block->local_blockid()] = overflow_free_;
      overflow_free_ = block->local_blockid();
    }
    block->value = ~new_owner;
    mutex_.Unlock();
    StopWrite(my_blockid);
  }
}

//----

char *DistributedCache::StartWrite(blockid_t blockid, bool is_partial) {
  mutex_.Lock();
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

char *DistributedCache::StartRead(BlockDevice::blockid_t blockid) {
  mutex_.Lock();
  BlockMetadata *block = &blocks_[blockid];
  if (likely(block->locks)) {
    block->locks++;
  } else {
    DecacheBlock_(blockid);
  }
  mutex_.Unlock();
  return block->data;
}

//----

void DistributedCache::StopRead(BlockDevice::blockid_t blockid) {
  mutex_.Lock();
  BlockMetadata *block = &blocks_[blockid];
  if (unlikely(--block->locks == 0)) {
    //fprintf(stderr, "Encaching\n");
    EncacheBlock_(blockid);
  }
  mutex_.Unlock();
}

void DistributedCache::StopWrite(BlockDevice::blockid_t blockid) {
  mutex_.Lock();
  BlockMetadata *block = &blocks_[blockid];
  if (unlikely(--block->locks == 0)) {
    //fprintf(stderr, "Encaching (WRITE)\n");
    EncacheBlock_(blockid);
  }
  mutex_.Unlock();
}

void DistributedCache::DecacheBlock_(BlockDevice::blockid_t blockid) {
  BlockMetadata *block = &blocks_[blockid];
  index_t slot = (unsigned(blockid) % unsigned(n_sets_)) << LOG_ASSOC;
  Slot *base_slot = &slots_[slot];

  DEBUG_ASSERT(!block->is_busy());
  block->locks = 1;

  if (likely(block->is_in_core())) {
    // It's in core, but its lock count was zero, so that means it's
    // definitely definitely in cache and in this line.
    for (int i = 0;; i++) {
      DEBUG_ASSERT_MSG(i != ASSOC, "Couldn't find %d in cache", blockid);
      if (unlikely(base_slot[i].blockid == blockid)) {
        base_slot[i].blockid = -1;
        break;
      }
    }
  } else {
    HandleMiss_(blockid);
  }
}

void DistributedCache::EncacheBlock_(BlockDevice::blockid_t blockid) {
  index_t slot = (unsigned(blockid) % unsigned(n_sets_)) << LOG_ASSOC;
  Slot *base_slot = &slots_[slot];
  int i;

  // Find first unused slot and move to front.
  i = 0;

  while (base_slot[i].blockid >= 0) {
    if (unlikely(i == ASSOC-1)) {
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

  base_slot->blockid = blockid;
}

//----

void DistributedCache::HandleMiss_(BlockDevice::blockid_t blockid) {
  BlockMetadata *block = &blocks_[blockid];

  DEBUG_ASSERT(block->data == NULL);
  block->data = mem::Alloc<char>(n_block_bytes_);
  DEBUG_ASSERT(block->data != NULL);

  if (block->is_new()) {
    DEBUG_ASSERT_MSG(block->status == NOT_DIRTY_NEW,
        "Block should be NOT_DIRTY_NEW, because that's what is_new() means");
    handler_->BlockInitFrozen(blockid, 0, n_block_bytes_, block->data);
  } else if (block->is_owner()) {
    DEBUG_ASSERT(block->status == NOT_DIRTY_OLD);
    HandleLocalMiss_(blockid);
  } else {
    DEBUG_ASSERT(block->status == NOT_DIRTY_OLD);
    HandleRemoteMiss_(blockid);
  }
  handler_->BlockThaw(blockid, 0, n_block_bytes_, block->data);
}

void DistributedCache::HandleLocalMiss_(BlockDevice::blockid_t blockid) {
  BlockMetadata *block = &blocks_[blockid];
  blockid_t local_blockid = block->local_blockid();

  DEBUG_ASSERT(block->is_owner());
  fprintf(stderr, "DISK: reading %d from %d\n", blockid, local_blockid);
  overflow_device_->Read(local_blockid,
      0, n_block_bytes_, block->data);
}

void DistributedCache::HandleRemoteMiss_(BlockDevice::blockid_t blockid) {
  BlockMetadata *block = &blocks_[blockid];

  DEBUG_ASSERT(!block->is_owner());
  ReadTransaction read_transaction;
  read_transaction.Doit(channel_num_, block->value,
      blockid, 0, n_block_bytes_, block->data);
}

void DistributedCache::Purge_(blockid_t blockid) {
  BlockMetadata *block = &blocks_[blockid];

  DEBUG_ASSERT(block->is_in_core());
  DEBUG_ASSERT_MSG(!block->is_busy(),
      "Trying to evict a busy block (non-zero lock count of %d)",
      int(block->locks));

  if (block->is_dirty()) {
    if (block->is_owner()) {
      WritebackDirtyLocalFreeze_(blockid);
    } else {
      WritebackDirtyRemote_(blockid);
    }
  }

  mem::Free(block->data);
  block->data = NULL;
}

void DistributedCache::WritebackDirtyLocalFreeze_(
    BlockDevice::blockid_t blockid) {
  BlockMetadata *block = &blocks_[blockid];

  DEBUG_ASSERT(block->is_owner());

  BlockDevice::blockid_t local_blockid = block->value;

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

  handler_->BlockFreeze(blockid, 0, n_block_bytes_, block->data, block->data);
  fprintf(stderr, "DISK: writing %d to %d\n", blockid, local_blockid);
  overflow_device_->Write(local_blockid, 0, n_block_bytes_, block->data);
  block->status = NOT_DIRTY_OLD;
}

void DistributedCache::WritebackDirtyRemote_(BlockDevice::blockid_t blockid) {
  BlockMetadata *block = &blocks_[blockid];

  DEBUG_ASSERT(!block->is_owner());
  DEBUG_ASSERT(block->is_dirty());

  if (block->status == FULLY_DIRTY) {
    // The entire block is dirty
    WriteTransaction write_transaction;
    write_transaction.Doit(channel_num_, block->value, blockid, handler_,
        0, n_block_bytes_, block->data);
  } else {
    DEBUG_ASSERT(block->status == PARTIALLY_DIRTY);
    // Find the intersection between this block and all dirty ranges we
    // know about.
    #ifdef DEBUG
    bool anything_done = false;
    #endif
    for (index_t i = 0; i < write_ranges_.size(); i++) {
      Position begin = write_ranges_[i].begin;
      Position end = write_ranges_[i].end;
      if (blockid >= begin.block || blockid <= end.block) {
        // We found a partial range that overlaps.  Write it.
        offset_t begin_offset = 0;
        offset_t end_offset = n_block_bytes_;
        if (blockid == begin.block) {
          begin_offset = begin.offset;
        }
        if (blockid == end.block) {
          end_offset = end.offset;
        }
        WriteTransaction write_transaction;
        write_transaction.Doit(channel_num_, block->value, blockid, handler_,
            begin_offset, end_offset, block->data);
        DEBUG_ONLY(anything_done = true);
      }
    }
    DEBUG_ASSERT_MSG(anything_done,
        "A block marked partially dirty has no overlapping write ranges.");
  }
  block->status = NOT_DIRTY_OLD;
}

void DistributedCache::AddPartialDirtyRange(
    blockid_t begin_block, offset_t begin_offset,
    blockid_t last_block, offset_t end_offset) {
  Position begin;
  Position end;
  begin.block = begin_block;
  begin.offset = begin_offset;
  end.block = last_block;
  end.offset = end_offset;
  write_ranges_.Union(begin, end);
}

//-------------------------------------------------------------------------
//-- PROTOCOL MESSAGES ----------------------------------------------------
//-------------------------------------------------------------------------

void DistributedCache::ReadTransaction::Doit(
    int channel_num, int peer, BlockDevice::blockid_t blockid,
    BlockDevice::offset_t begin, BlockDevice::offset_t end,
    char *buffer) {
  Transaction::Init(channel_num);
  Message *message = CreateMessage(peer, sizeof(Request));
  Request *request = message->data_as<Request>();
  request->type = Request::READ;
  request->blockid = blockid;
  request->begin = begin;
  request->end = end;
  request->rank = 0;
  response = NULL;
  Send(message);
  cond.Wait();
  mem::Copy(buffer, response->data(), end - begin);
  delete response;
}

void DistributedCache::ReadTransaction::HandleMessage(Message *message) {
  response = message;
  cond.Done();
  Done();
}

//-------------------------------------------------------------------------

BlockDevice::blockid_t DistributedCache::AllocTransaction::Doit(
    int channel_num, int peer, blockid_t n_blocks_to_alloc, int owner) {
  Transaction::Init(channel_num);
  Message *message = CreateMessage(peer, sizeof(Request));
  Request *request = message->data_as<Request>();
  request->type = Request::ALLOC;
  request->blockid = n_blocks_to_alloc;
  request->begin = 0;
  request->end = 0;
  request->rank = owner;
  response = NULL;
  Send(message);
  cond.Wait();
  blockid_t retval = *message->data_as<blockid_t>();
  delete response;
  return retval;
}

void DistributedCache::AllocTransaction::HandleMessage(Message *message) {
  response = message;
  cond.Done();
  Done();
}

//-------------------------------------------------------------------------

void DistributedCache::WriteTransaction::Doit(
    int channel_num, int peer,
    BlockDevice::blockid_t blockid, BlockHandler *handler,
    BlockDevice::offset_t begin, BlockDevice::offset_t end,
    const char *buffer) {
  Transaction::Init(channel_num);
  Message *message = CreateMessage(peer, Request::size(end - begin));
  Request *request = message->data_as<Request>();
  request->type = Request::WRITE;
  request->blockid = blockid;
  request->begin = begin;
  request->end = end;
  request->rank = 0;
  mem::Copy(request->data_as<char>(), buffer, end - begin);
  handler->BlockFreeze(blockid, begin, end, buffer, request->data_as<char>());
  Send(message);
  Done();
  // no wait necessary
}

void DistributedCache::WriteTransaction::HandleMessage(Message *message) {
  FATAL("No response to DistributedCache::WriteTransaction expected");
}

//-------------------------------------------------------------------------

DistributedCache::ConfigResponse *DistributedCache::ConfigTransaction::Doit(
    int channel_num, int peer) {
  Transaction::Init(channel_num);
  Message *message = CreateMessage(peer, sizeof(Request));
  Request *request = message->data_as<Request>();
  request->type = Request::CONFIG;
  request->blockid = 0;
  request->begin = 0;
  request->end = 0;
  request->rank = 0;
  Send(message);
  cond.Wait();
  return ot::PointerThaw<ConfigResponse>(response->data());
}

void DistributedCache::ConfigTransaction::HandleMessage(Message *message) {
  response = message;
  cond.Done();
  Done();
}

//-------------------------------------------------------------------------

void DistributedCache::ResponseTransaction::Init(
    DistributedCache *cache_in) {
  cache_ = cache_in;
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
      Message *response = CreateMessage(
          message->peer(), request->end - request->begin);
      cache_->RemoteRead(request->blockid, request->begin, request->end,
          response->data());
      Send(response);
    }
    break;
    case Request::WRITE: {
      cache_->RemoteWrite(request->blockid, request->begin, request->end,
          request->data_as<char>());
    }
    break;
    case Request::ALLOC: {
      DEBUG_ASSERT(cache_->my_rank_ == MASTER_RANK);
      Message *response = CreateMessage(message->peer(), sizeof(blockid_t));
      *message->data_as<blockid_t>() =
          cache_->AllocBlocks(request->blockid, request->rank);
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
  cache.ComputeStatusInformation_(&statuses);
}

void DistributedCache::SyncInfo::MergeWith(const SyncInfo& other) {
  index_t old_size = statuses.size();

  for (index_t i = 0; i < old_size; i++) {
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
  mutex_.Unlock();
  delete message;
  if (state_ == DONE) {
    delete this;
  }
}

void DistributedCache::SyncTransaction::StartSyncFlushDone() {
  mutex_.Lock();
  ChildFlushed_();
  mutex_.Unlock();
  if (state_ == DONE) {
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
  SyncInfo *info = ot::PointerThaw<SyncInfo>(message->data());
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
      SendStatusInformation_(rpc::parent());
    }
  }
}

void DistributedCache::SyncTransaction::ParentAccumulated_(
    const SyncInfo& info) {
  cache_->channel_.SyncDone();
  cache_->HandleSyncInfo_(info);
  for (index_t i = 0; i < rpc::n_children(); i++) {
    SendStatusInformation_(rpc::child(i));
  }
  Done();
  state_ = DONE;
}

void DistributedCache::SyncTransaction::SendBlankSyncMessage_(int peer) {
  Message *request_msg = CreateMessage(peer, sizeof(Request));
  Request *request = request_msg->data_as<Request>();
  request->type = Request::SYNC;
  request->blockid = 0;
  request->begin = 0;
  request->end = 0;
  Send(request_msg);
}

void DistributedCache::SyncTransaction::SendStatusInformation_(int peer) {
  // we have two layers of headers here, and then we can freeze the
  // ArrayList into place.
  Message *request_msg = CreateMessage(peer, Request::size(
      ot::PointerFrozenSize(sync_info_)));
  Request *request = request_msg->data_as<Request>();
  ot::PointerFreeze(sync_info_, request->data_as<char>());
  request->type = Request::SYNC;
  request->blockid = 0;
  request->begin = 0;
  request->end = 0;
  Send(request_msg);
}

//-------------------------------------------------------------------------

void DistributedCache::CacheChannel::Init(DistributedCache *cache_in) {
  cache_ = cache_in;
  sync_transaction_ = NULL;
}

void DistributedCache::CacheChannel::StartSyncFlushDone() {
  SyncTransaction *t;
  mutex_.Lock();
  if (sync_transaction_ == NULL) {
    sync_transaction_ = new SyncTransaction();
    sync_transaction_->Init(cache_);
  }
  t = sync_transaction_;
  mutex_.Unlock();
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

Transaction *DistributedCache::CacheChannel::GetTransaction(
    Message *message) {
  Request *request = reinterpret_cast<Request*>(message->data());
  if (unlikely(request->type == Request::SYNC)) {
    Transaction *t;
    mutex_.Lock();
    if (sync_transaction_ == NULL) {
      sync_transaction_ = new SyncTransaction();
      sync_transaction_->Init(cache_);
    }
    t = sync_transaction_;
    mutex_.Unlock();
    return t;
  } else {
    ResponseTransaction *t = new ResponseTransaction();
    t->Init(cache_);
    return t;
  }
}

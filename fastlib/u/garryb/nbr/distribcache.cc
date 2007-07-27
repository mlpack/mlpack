/*
to-do

- Init functions
- the protocol
  - nothing is initialized in my version yet
  - alloc() ?
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
}

void DistributedCache::InitFile_(const char *filename) {
  String filename_str;
  if (filename == NULL) {
    filename_str.Copy(tmpnam(NULL));
  } else {
    filename_str.Copy(filename);
  }
  overflow_device_ = new DiskBlockDevice(filename_str.c_str(),
      BlockDevice::M_TEMP, n_block_bytes_);
}

void DistributedCache::InitChannel_(int channel_num_in) {
  channel_num_ = channel_num_in;
  channel_.Init(this);
}

void DistributedCache::InitCommon_(
    BlockDevice::offset_t n_block_bytes) {
  blocks_.Init();
  handler_ = NULL;

  overflow_free_ = -1;
  overflow_metadata_.Init();
  overflow_device_ = NULL;

  my_rank_ = rpc::Rank();
}

void DistributedCache::InitCache_(size_t total_ram) {
  ASSOC = 8;
  LOG_ASSOC = 3;
  // give enough cache sets, but rounded up
  n_sets_ = (total_ram + ASSOC*n_block_bytes_ - 1) / (ASSOC*n_block_bytes_);
  slots_.Init(n_sets_ << LOG_ASSOC);
}

void DistributedCache::HandleStatusInformation_(
    const ArrayList<BlockStatus>& statuses) {
  // This method is only called after a sync.
  // However, it is possible that some other machines might have started
  // writing stuff, so we'll have to take this information with a grain of
  // salt.

  mutex_.Lock();

  DEBUG_ASSERT(statuses.size() >= n_blocks_);

  if (n_blocks_ != statuses.size()) [
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

  mutex_.Unlock();
}

#error need GiveOwnership method that marks entire page FULLY_DIRTY

void DistributedCache::ComputeStatusInformation_(
    ArrayList<BlockStatus> *statuses)
  mutex_.Lock();
  statuses->Init(cache_->n_blocks());
  for (index_t i = 0; i < statuses->size(); i++) {
    BlockStatus *status = &statuses[i];
    BlockMetadata *owner = &cache_->blocks_[i];
    if (block->is_owner()) {
      status->owner = rank_;
    } else if (self_only) {
      status->owner = -1;
    }
    status->is_new = block->is_new();
  }
  mutex_.Unlock();
}

void DistributedCache::BestEffortFlush(double portion) {
  mutex_.Lock();
  Slot *slot = slots_.ptr();
  index_t i = slots_.size();
  int start_col = int(nearbyint(ASSOC * portion));
  BlockMetadata *blocks = blocks_.ptr();

  // Might want to software-pipeline this loop, because of the really nasty
  // indirect load going on.
  do {
    i -= ASSOC;
    for (int j = start_col; j < ASSOC; j++) {
      blockid_t blockid = slot[j].blockid;
      if (blockid >= 0) {
        BlockMetadata *block = &blocks[blockid];
        if (unlikely(block->is_dirty()) && unlikely(!block->is_owner())) {
          DEBUG_ASSERT_MSG(block->locks == 0, "Why is a locked block in LRU?");
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
  Slot *slot = slots_.ptr();
  index_t i = slots_.size();
  BlockMetadata *blocks = blocks_.ptr();

  // Might want to software-pipeline this loop, because of the really nasty
  // indirect load going on.
  do {
    i--;
    blockid_t blockid = slot->blockid;
    BlockMetadata *block = &blocks[blockid];
    slot++;
    if (unlikely(!block->is_owner())) {
      slot->blockid = -1;
      DEBUG_ASSERT_MSG(block->locks == 0, "Why is a locked block in LRU?");
      Purge_(blockid);
    }
  } while (i != 0);
  mutex_.Unlock();

  // TODO: Make absolutely certain nobody is currently accessesing the cache
  write_ranges_.Clear();

  channel_.StartSyncFlushDone();
}

void DistributedCache::WaitSync() {
  channel_.WaitSync();
}

//----

void DistributedCache::Read(blockid_t blockid,
    offset_t begin, offset_t end, char *buf) {
  const char *src = StartRead(blockid);
  size_t n_bytes = end - begin;

  // TODO: consider read-through
  mem::Copy(buf, src + begin, n_bytes);
  block_handler_->BlockFreeze(blockid, begin, n_bytes, src + begin, buf);

  StopRead(blockid);
}

void DistributedCache::Write(blockid_t blockid,
    offset_t begin, offset_t end, const char *buf) {
  char *dest = StartWrite(blockid);
  size_t n_bytes = end - begin;

  mem::CopyBytes(dest + begin, buf, n_bytes);
  block_handler_->BlockThaw(blockid, begin, n_bytes, dest + begin);

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
    int32 value = (owner == my_rank_) ? SELF_OWNER_UNALLOCATED : (~owner);
    for (blockid_t i = 0; i < n_blocks_to_alloc; i++) {
      block[i].value = value;
    }
    mutex_.Unlock();
  } else {
    AllocTransacation t;
    blockid = t.Doit(channel_, MASTER_RANK, n_blocks_to_alloc, owner);
    mutex_.Lock();
    n_blocks_ = blockid + n_blocks_to_alloc;
    blocks_.GrowTo(n_blocks_);
    mutex_.Unlock();
  }

  return blockid;
}

//----

char *DistributedCache::StartWrite(BlockDevice::blockid blockid,
    bool partial) {
  mutex_.Lock();
  BlockMetadata *block = &blocks_[blockid];
  if (likely(block->locks)) {
    block->locks++;
  } else {
    DecacheBlock_(blockid);
  }
  if (partial) {
    block->status &= PARTIALLY_DIRTY;
  } else {
    block->status = FULLY_DIRTY;
  }
  mutex_.Unlock();
  return block->data;
}

char *DistributedCache::StartRead(BlockDevice::blockid blockid) {
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
  if (unlikely(--block->locks == 0)) {
    EncacheBlock_(blockid);
  }
  mutex_.Unlock();
}

void DistributedCache::StopWrite(BlockDevice::blockid_t blockid) {
  mutex_.Lock();
  if (unlikely(--block->locks == 0)) {
    EncacheBlock_(blockid);
  }
  mutex_.Unlock();
}

void DistributedCache::DecacheBlock_(BlockDevice::blockid_t blockid) {
  BlockMetadata *block = &blocks_[blockid];
  index_t slot = (unsigned(blockid) % unsigned(n_sets_)) << LOG_ASSOC;
  Slot *base_slot = &slots_[slot];

  DEBUG_ASSERT(block->locks == 0);
  block->locks = 1;

  if (likely(block->is_in_core())) {
    // It's in core, but its lock count was zero, so that means it's
    // definitely definitely in cache and in this line.
    for (int i = 0;; i++) {
      if (unlikely(base_slot[i].blockid == blockid)) {
        base_slot[i].blockid = -1;
        break;
      }
      DEBUG_ASSERT(i != ASSOC);
    }
  } else {
    HandleMiss_(blockid);
  }
}

void DistributedCache::EncacheBlock_(BlockDevice::blockid_t blockid) {
  BlockMetadata *block = &blocks_[blockid];
  index_t slot = (unsigned(blockid) % unsigned(n_sets_)) << LOG_ASSOC;
  Slot *base_slot = &slots_[slot];
  int i;

  // Find first unused slot and move to front.
  i = 0;

  while (base_slot[i].blockid < 0) {
    if (unlikely(++i == ASSOC)) {
      Purge_(base_slot->blockid);
      i--;
      break;
    }
  }

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
    block_handler_->BlockInitFrozen(blockid, 0, n_block_bytes_, block->data);
    block_handler_->BlockThaw(blockid, 0, n_block_bytes_, block->data);
  } else if (block->is_owner()) {
    DEBUG_ASSERT(block->status == NOT_DIRTY_OLD);
    HandleLocalMiss_(blockid);
  } else {
    DEBUG_ASSERT(block->status == NOT_DIRTY_OLD);
    HandleRemoteMiss_(blockid);
  }
}

void DistributedCache::HandleLocalMiss_(BlockDevice::blockid_t blockid) {
  BlockMetadata *block = &blocks_[blockid];
  blockid_t local_blockid = block->local_blockid();

  DEBUG_ASSERT(block->is_owner());
  fprintf(stderr, "DISK: Reading block %d from %d\n", blockid, local_blockid);
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

char *DistributedCache::Purge_(BlockDevice::blockid blockid) {
  BlockMetadata *block = &blocks_[blockid];

  DEBUG_ASSERT(block->is_in_core());
  DEBUG_ASSERT_MSG(!block->is_busy(),
      "Trying to evict a busy block (non-zero lock count of %d)",
      int(block->locks));

  if (block->is_dirty()) {
    if (block->is_owner()) {
      PurgeDirtyLocal_(blockid);
      return;
    } else {
      WritebackDirtyRemote_(blockid);
    }
  }

  mem::Free(block->data);
  block->data = NULL;
}

void DistributedCache::PurgeDirtyLocal_(BlockDevice::blockid_t blockid) {
  BlockMetadata *block = &blocks_[blockid];

  DEBUG_ASSERT(block->is_owner());
  DEBUG_ASSERT_MSG(block->value == SELF_OWNER_UNALLOCATED,
      "Blocks that are written back locally are purged.");
  // First, inore the block's actual block ID, it's not important -- we
  // assign these only when we write them back.
  // Second, always write back the entire block.  Since we only assign block
  // ID's at writeback time, we're overwriting completely unrelated data :-)

  BlockDevice::blockid_t local_blockid = block->value;

  if (block->value == SELF_OWNER_UNALLOCATED) {
    local_blockid = overflow_free_;
    if (local_blockid < 0) {
      local_blockid = overflow_device_->AllocBlocks(1);
    } else {
      // TODO: The free list is not used in the current code, but it would
      // become useful if it were possible for data to move dynamically.
      overflow_free_ = overflow_next_[local_blockid];
    }
    block->value = local_blockid;
  }

  fprintf(stderr, "DISK: Writing block %d to %d\n", blockid, local_blockid);
  local_device_.Write(local_blockid, 0, n_block_bytes_, block->data);
  block->status = NOT_DIRTY_OLD;

  mem::Free(block->data);
  block->data = NULL;
}

void DistributedCache::WritebackDirtyRemote_(BlockDevice::blockid_t blockid) {
  BlockMetadata *block = &blocks_[blockid];

  DEBUG_ASSERT(!block->is_owner());
  DEBUG_ASSERT(block->is_dirty());

  if (block->status == FULLY_DIRTY) {
    // The entire block is dirty
    WriteTransaction write_transaction;
    write_transaction.Doit(channel_num, block->value, blockid,
        0, n_block_bytes_, block->data);
  } else {
    DEBUG_ASSERT(block->status == PARTIALLY_DIRTY);
    // Find the intersection between this block and all dirty ranges we
    // know about.
    #ifdef DEBUG
    bool anything_done = false;
    #endif
    for (index_t i = 0; i < ranges_.size(); i++) {
      const Range *range = &ranges_[i];
      if (block >= range->begin_block || block <= range->last_block) {
        // We found a partial range that overlaps.  Write it.
        offset_t begin = 0;
        offset_t end = n_block_bytes_;
        if (block == range->begin_block) {
          begin = range->begin;
        }
        if (block == range->last_block) {
          end = range->end;
        }
        WriteTransaction write_transaction;
        write_transaction.Doit(channel_num, block->value, blockid,
            begin, end, block->data);
        DEBUG_ONLY(anything_done = true);
      }
    }
    DEBUG_ASSERT_MSG(anything_done,
        "A block marked partially dirty has no overlapping write ranges.");
  }
  block->status = NOT_DIRTY_OLD;
}

void AddPartialDirtyRange(blockid_t begin_block, offset_t begin_offset,
    blockid_t last_block, offset_t end_offset) {
  // This is a range-merge algorithm.  I tried to make it very simple --
  // it's not very efficient, but this function is very rarely called.
  ArrayList<Range> new_list;
  index_t i;
  Range new_range;

  new_range.begin_block = begin_block;
  new_range.begin = begin_offset;
  new_range.last_block = last_block;
  new_range.end_block = end_block;

  new_list.Init();
  
  i = 0;

  // add everything that strictly precedes the new one to add
  while (i < ranges_.size() && !new_range.BeginsBeforeEnd(ranges_[i])) {
    *new_list.AddBack() = ranges_[i];
    i++;
  }

  // merge all ranges that overlap into this range
  while (i < ranges_.size() && ranges_[i].BeginsBeforeEnd(new_range)) {
    new_range.Merge(ranges_[i]);
    i++;
  }

  // add the resulting range
  *new_list.AddBack() = new_range;

  // add everything that comes after
  for (; i < ranges_.size(); j++) {
    *new_list.AddBack() = ranges_[i];
  }

  // replace the list
  ranges_.Swap(&new_list);
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

void DistributedCache::WriteTransaction::Doit(
    int channel_num, int peer, BlockDevice::blockid_t blockid,
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
      cache_->handler_->Serialize(&config_response.data);
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
      cache_->Read(request->blockid, request->begin, request->end,
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
      *reinterpret_cast<blockid_t>(message->data()) =
          cache_->AllocBlocks(blockid, request->rank);
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

void DistributedCache::SyncTransaction::Init(Cache *cache) {
  cache_ = cache_in;
  state_ = CHILDREN_FLUSHING;
  n_ = 0;
  sync_response_.Init();
}

void DistributedCache::SyncTransaction::HandleMessage(Message *message) {
  mutex_.Lock();
  switch (state_) {
    case CHILDREN_FLUSHING:
      ChildFlushed_();
      break;
    case OTHERS_FLUSHING:
      ParentFlushed_();
      break;
    case CHILDREN_ACCUMULATING:
      AccumulateChild_(message);
      break;
    case OTHERS_ACCUMULATING: {
      DEBUG_ASSERT(message->peer() == rpc::parent());
      ArrayList<BlockStatus> *in_statuses =
          ot::PointerThaw< ArrayList<BlockStatus> >(
              message->data_as<Request>()->data_as<char>());
      ParentAccumulated_(*in_statuses);
    }
    break;
    default:
      FATAL("Unknown state");
  }
  mutex_.Unlock();
  delete message;
}

void DistributedCache::SyncTransaction::StartSyncFlushDone() {
  mutex_.Lock();
  ChildFlushed_();
  mutex_.Unlock();
}

void DistributedCache::SyncTransaction::ChildFlushed_() {
  n++;
  // must accumulated n_children + 1: myself!
  if (n == rpc::n_children() + 1) {
    if (rpc::is_root()) {
      ParentFlushed_();
    } else {
      state_ = OTHERS_FLUSHING;
      SendBlankSyncMessage_(rpc::parent());
    }
  }
}

void DistributedCache::SyncTransaction::ParentFlushed_() {
  DEBUG_ASSERT(message->peer() == rpc::parent());
  n = 0;
  for (index_t i = 0; i < rpc::n_children(); i++) {
    SendBlankSyncMessage_(rpc::child(i));
  }
  // Now we are absolutely certain ALL machines have finished flushing.
  // Thus, our block ownership is accurate.
  cache_->ComputeStatusInformation_(&statuses);
  state_ = CHILDREN_ACCUMULATING;
  CheckAccumulation_();
}

void DistributedCache::SyncTransaction::AccumulateChild_(Message *message) {
  index_t old_size = statuses_.size();
  for (index_t i = 0; i < old_size; i++) {
    BlockStatus *orig = &statuses_[i];
    BlockStatus *in = &(*in_statuses)[i];
    if (in->owner >= 0) {
      DEBUG_ASSERT(orig->owner < 0);
      *orig = *in;
    }
  }
  if (old_size < in_statuses->size()) {
    statuses_.Resize(in_statuses->size());
    mem::Copy(&statuses_[old_size], &(*in_statuses)[old_size],
        statuses_.size() - old_size);
  }
  n++;
  CheckAccumulation_();
}

void DistributedCache::SyncTransaction::CheckAccumulation_() {
  // only need to accumulate children, since i myself am a given
  if (n == rpc::n_children()) {
    if (rpc::is_root()) {
      ParentAccumulated_(statuses_);
    } else {
      state_ = OTHERS_ACCUMULATING;
      SendStatusInformation_(rpc::parent());
    }
  }
}

void DistributedCache::SyncTransaction::ParentAccumulated_(const ArrayList<BlockStatus&> statuses_in) {
  cache_->channel_.SyncDone();
  cache_->HandleStatusInformation_(statuses_in);
  for (index_t i = 0; i < rpc::n_children(); i++) {
    SendStatusInformation_(rpc::child(i));
  }
  Done();
}

void DistributedCache::SyncTransaction::SendBlankSyncMessage_(int peer) {
  Message *request_msg = CreateMessage(child, sizeof(Request));
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
  Message *request_msg = CreateMessage(child, Request::size(
      ot::PointerFrozenSize(statuses_)));
  Request *request = request_msg->data_as<Request>();
  ot::PointerFreeze(statuses_, request->data_as<char>());
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
  mutex_.Lock();
  if (sync_transaction_ == NULL) {
    sync_transaction_ = new SyncTransaction();
    sync_transaction_->Init(cache_);
  }
  sync_transaction_->StartSyncFlushDone();
  mutex_.Unlock();
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

/*
  // TODO: This method is kind of complicated and is probably slow.
  // Can we make it simpler?  The good side is, however, that the FIFO
  // cache_ probably won't call this too often (since, well, it is a cache_!)
  int rangeid = block->dirty_ranges;

#error this range stuff is too complicated, fix it

  if (unlikely(rangeid < 0)) {
    if (block->is_owner()) {
      block->dirty_ranges = FULLY_DIRTY;
    } else if (end - begin == n_block_bytes_) {
      MarkDirty_(block);
    } else {
      rangeid = free_range_;

      if (unlikely(rangeid < 0)) {
        rangeid = write_ranges_.size();
        ranges = write_ranges_.AddBack();
        block->dirty_ranges = rangeid;
      } else {
        ranges = &write_ranges_[rangeid];
        free_range_ = rangeid;
        block->dirty_ranges = rangeid;
      }

      Range *range = ranges->ranges.AddBack();
      range->begin = begin;
      range->end = end;
    }
  } else {
    // Try to merge the range, or add it to the list
    Range *first_range = ranges->ranges.ptr();
    index_t size = ranges->ranges.size();

    for (index_t i = 0; i < size; i++) {
      range = &first_range[i];
      if (begin <= range->end) {
        if (end >= range->begin) {
          range->begin = min(begin, range->begin);
          range->end = max(end, range->end);
          return;
        } else {
          range = ranges->ranges.AddBack();
          for (index_t j = ranges->ranges.size() - i - 1; j != 0; j--) {
            range[0] = range[-1];
            range--;
          }
          range->begin = begin;
          range->end = end;
          return;
        }
      }
  }
*/
//    // Flush each independent dirty range
//    WriteRanges *ranges = &ranges_[rangeid];
//    offset_t begin = 0;
//    for (index_t i = 0; i < ranges->ranges.size(); i++) { 
//      Range *range = &ranges->ranges[i];
//      WriteTransaction write_transaction;
//      begin = max(begin, range->begin);
//      if (begin != range->end) {
//        write_transaction.Doit(channel_num, block->value, blockid,
//            begin, range->end, block->data + range->begin);
//      }
//      begin = max(begin, range->end);
//    }
//    ranges->next = free_range_;
//    ranges->ranges.Resize(0);
//    free_range_ = rangeid;

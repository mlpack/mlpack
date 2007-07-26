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
  InitCache_(total_ram);
  InitChannel_(channel_num_in);
}

void DistributedCache::InitChannel_(int channel_num_in) {
  channel_num_ = channel_num_in;
  channel_.cache = cache;
}

void DistributedCache::InitCommon_(
    BlockDevice::offset_t n_block_bytes) {
  blocks_.Init();
  handler_ = NULL;

  //write_ranges_.Init();

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
  DEBUG_ASSERT(statis.size() >= n_blocks_);

  if (n_blocks_ != statuses.size()) [
    n_blocks_ = statuses.size();
    blocks_.Resize(n_blocks_);
  }

  for (index_t i = 0; i < n_blocks_; i++) {
    BlockMetadata *block = &blocks_[i];
    const BlockStatus *status = &statuses[i];
      block->data = NULL;

    block->data = NULL;
    block->locks = 0;
    if (status->owner == my_rank_) {
      block->value = SELF_OWNER_UNALLOCATED;
    } else {
      block->value = ~status->owner;
    }
    if (!block->is_dirty()) {
      block->status = status->is_new ? NOT_DIRTY_NEW : NOT_DIRTY_OLD;
    }
  }
}

void DistributedCache::BestEffortFlush() {
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
    if (unlikely(block->is_dirty()) && unlikely(!block->is_owner())) {
      DEBUG_ASSERT_MSG(block->locks == 0, "Why is a locked block in LRU?");
      WritebackDirtyRemote_(blockid);
    }
    slot++;
  } while (i != 0);
  mutex_.Unlock();
}

void DistributedCache::PreBarrierSync() {
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
}

void DistributedCache::PostBarrierSync() {
  if (my_rank_ != MASTER_RANK) {
    QueryTransaction qt;
    QueryResponse *response = qt.Doit(channel_num_in, MASTER_RANK);
    mutex_.Lock();
    HandleStatusInformation_(response->statuses);
    mutex_.Unlock();
  }
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

  for (;;) {
    if (unlikely(base_slot[i].blockid < 0)) {
      break;
    }
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

  DEBUG_ASSERT(block->is_owner());
  overflow_device_->Read(block->local_blockid(),
      0, n_block_bytes_, block->data);
  overflow_metadata_[block->local_blockid()].next_free_local = overflow_free_;
  overflow_free_ = block->local_blockid();
}

void DistributedCache::HandleRemoteMiss_(BlockDevice::blockid_t blockid) {
  BlockMetadata *block = &blocks_[blockid];

  DEBUG_ASSERT(!block->is_owner());
  ReadTransaction read_transaction;
  read_transaction.Doit(channel_num_, block->value,
      blockid, 0, n_block_bytes_, block->data);
}

#warning "consider making encache and decache fast for the first block"

char *DistributedCache::Purge_(BlockDevice::blockid blockid) {
  BlockMetadata *block = &blocks_[blockid];

  DEBUG_ASSERT(block->is_in_core());
  DEBUG_ASSERT_MSG(!block->is_busy(), "Trying to evict a busy block");

  if (block->is_dirty()) {
    if (block->is_owner()) {
      WritebackDirtyLocal_(blockid);
    } else {
      WritebackDirtyRemote_(blockid);
    }
  }

  mem::Free(block->data);
  block->data = NULL;
}

void DistributedCache::WritebackDirtyLocal_(BlockDevice::blockid_t blockid) {
  BlockMetadata *block = &blocks_[blockid];

  DEBUG_ASSERT(block->is_owner());
  //DEBUG_ASSERT_MSG(block->status == FULLY_DIRTY,
  //    "Local blocks should only be fully dirty or not dirty at all");
  //  local blocks can be fully or partially dirty, we simply don't care
  // Ignore the block's actual block ID, it's not important -- we assign
  // these only when we write them back.

  BlockDevice::blockid_t local_blockid = overflow_free_;

  if (local_blockid < 0) {
    local_blockid = overflow_device_->AllocBlocks(1);
    overflow_metadata_.Resize(local_blockid + 1);
  }

  block->value = local_blockid;

  local_device_.Write(local_blockid, 0, n_block_bytes_, block->data);
  block->status = NOT_DIRTY_OLD;
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
    #error compute range stuff
  }
  block->status = NOT_DIRTY_OLD;
}

//-------------------------------------------------------------------------
//-- PROTOCOL MESSAGES ----------------------------------------------------
//-------------------------------------------------------------------------

/** Net-transferable request operation */
struct DCRequest {
 public:
  enum { CONFIG, QUERY, READ, WRITE, ALLOC } type;
  BlockDevice::blockid_t blockid;
  BlockDevice::offset_t begin;
  BlockDevice::offset_t end;
  int rank;
};

/** Net-transferable write operation */
struct DCWriteRequest : public DCRequest {
  long long_data[1];

  char *data() {
    return reinterpret_cast<char*>(long_data);
  }

  static size_t size(size_t data_size) {
    return sizeof(DCWriteRequest) + data_size - sizeof(long_data);
  }
};

//-------------------------------------------------------------------------

void DistributedCache::ReadTransaction::Doit(
    int channel_num, int peer, BlockDevice::blockid_t blockid,
    BlockDevice::offset_t begin, BlockDevice::offset_t end,
    char *buffer) {
  Transaction::Init(channel_num);
  Message *message = CreateMessage(peer, sizeof(DCRequest));
  DCRequest *request = reinterpret_cast<DCRequest*>(message->data());
  request->type = DCRequest::READ;
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
  Message *message = CreateMessage(peer, DCWriteRequest::size(end - begin));
  DCWriteRequest *request = reinterpret_cast<DCWriteRequest*>(message->data());
  request->type = DCRequest::WRITE;
  request->blockid = blockid;
  request->begin = begin;
  request->end = end;
  request->rank = 0;
  mem::Copy(request->data(), response->data(), end - begin);
  Send(message);
  Done();
  // no wait necessary
}

void DistributedCache::WriteTransaction::HandleMessage(Message *message) {
  FATAL("No response to DistributedCache::WriteTransaction expected");
}

//-------------------------------------------------------------------------

DistributedCache::ConfigResponse *r = DistributedCache::ConfigTransaction::Doit(
    int channel_num, int peer) {
  Transaction::Init(channel_num);
  Message *message = CreateMessage(peer, sizeof(DCRequest));
  DCRequest *request = reinterpret_cast<DCRequest*>(message->data());
  request->type = DCRequest::CONFIG;
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

DistributedCache::QueryResponse *r = DistributedCache::QueryTransaction::Doit(
    int channel_num, int peer) {
  Transaction::Init(channel_num);
  Message *message = CreateMessage(peer, sizeof(DCRequest));
  DCRequest *request = reinterpret_cast<DCRequest*>(message->data());
  request->type = DCRequest::QUERY;
  request->blockid = 0;
  request->begin = 0;
  request->end = 0;
  request->rank = 0;
  Send(message);
  cond.Wait();
  return ot::PointerThaw<QueryResponse>(response->data());
}

void DistributedCache::QueryTransaction::HandleMessage(Message *message) {
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
  DCRequest *request = reinterpret_cast<DCRequest*>(message->data());

  switch (request->type) {
    case DCRequest::CONFIG: {
      DCConfigResponse config_response;
      cache_->handler_->Serialize(&config_response.data);
      config_response.n_block_bytes = cache_->n_block_bytes_;
      Message *response = CreateMessage(message->peer(),
          ot::PointerFrozenSize(config_response));
      ot::PointerFreeze(config_response, response->data());
      Send(response);
    }
    case DCRequest::QUERY: {
      DCQueryResponse query_response;
      query_response.statuses.Init(cache_->n_blocks());
      for (index_t i = 0; i < query_response.statuses.size(); i++) {
        BlockStatus *status = &query_response.statuses[i];
        BlockMetadata *owner = &cache_->blocks_[i];
        status->owner = block->owner(cache_);
        status->is_new = block->is_new();
      }
      Message *response = CreateMessage(message->peer(),
          ot::PointerFrozenSize(query_response));
      ot::PointerFreeze(query_response, response->data());
      Send(response);
    }
    break;
    case DCRequest::READ: {
      Message *response = CreateMessage(
          message->peer(), request->end - request->begin);
      cache_->Read(request->blockid, request->begin, request->end,
          response->data());
      Send(response);
    }
    break;
    case DCRequest::WRITE: {
      cache_->RemoteWrite(request->blockid, request->begin, request->end,
          static_cast<DCWriteRequest*>(request)->data());
    }
    case DCRequest::ALLOC: {
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

Transaction *DistributedCache::CacheChannel::GetTransaction(
    Message *message) {
  DCResponseTransaction *t = new DCResponseTransaction();
  t->Init(cache);
  return t;
}

/*
  // TODO: This method is kind of complicated and is probably slow.
  // Can we make it simpler?  The good side is, however, that the FIFO
  // cache probably won't call this too often (since, well, it is a cache!)
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

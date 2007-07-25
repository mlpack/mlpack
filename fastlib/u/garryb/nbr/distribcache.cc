//-------------------------------------------------------------------------
//-- THE DISTRIBUTED CACHE ------------------------------------------------
//-------------------------------------------------------------------------

void DistributedCache::InitMaster(int channel_num_in,
    BlockDevice::blockid_t n_block_bytes_in,
    BlockHandler *handler_in) {
  n_blocks_ = 0;
  n_block_bytes_ = n_block_bytes_in;
  handler_ = handler_in;
  InitChannel_(channel_num_in);
}

void DistributedCache::InitChannel(int channel_num_in) {
  channel_num_ = channel_num_in;
  channel_.cache = cache;
}

void DistributedCache::InitWorker(int channel_num) {
  // connect to master and figure out specs
}

void DistributedCache::HandleStatusInformation_(
    BlockDevice::blockid_t blockid, const BlockStatus& status) {
  BlockMetadata *block = &blocks_[blockid];
  if (unlikely(status.owner == my_rank_)) {
    if (block->is_owner) {
      // was owner, still am owner
      return;
    } else {
      block->pointer = NULL;
      block->state = NEW;
      block->value = -1; /* Not in cache */
    }
  } else {
    block->value = status.owner;
  }
}

char *DistributedCache::StartWrite(BlockDevice::blockid blockid,
    BlockDevice::offset_t begin_offset, BlockDevice::offset_t end_offset) {
  BlockMetadata *block = &blocks_[blockid];
  MarkUse_(blockid);
  block->is_new = false;
  return block->data;
}

char *DistributedCache::StartRead(BlockDevice::blockid blockid) {
  BlockMetadata *block = &blocks_[blockid];
  MarkUse_(blockid);
  return block->data;
}

void DistributedCache::MarkUse_(BlockDevice::blockid blockid) {
  MarkUse_(blockid);
}

void DistributedCache::HandleMiss_(BlockDevice::blockid_t blockid) {
  BlockMetadata *block = &blocks_[blockid];

  DEBUG_ASSERT(block->data == NULL);
  block->data = mem::Alloc<char*>(n_block_bytes_);

  if (likely(block->is_new)) {
    block_handler_->BlockInitFrozen(blockid, 0, n_block_bytes_, block->data);
    block_handler_->BlockThaw(blockid, 0, n_block_bytes_, block->data);
  } else if (block->is_owner) {
    HandleLocalMiss_(blockid);
  } else {
    HandleRemoteMiss_(blockid);
  }
  block->dirty_ranges = NOT_DIRTY;
}

void DistributedCache::HandleLocalMiss_(BlockDevice::blockid_t blockid) {
  BlockMetadata *block = &blocks_[blockid];

  DEBUG_ASSERT(block->is_owner);
  block->data = mem::Alloc<char*>(n_block_bytes_);
  DEBUG_ASSERT(block->value >= 0);
  overflow_device_->Read(block->value, 0, n_block_bytes_, block->data);
}

void DistributedCache::HandleRemoteMiss_(BlockDevice::blockid_t blockid) {
  BlockMetadata *block = &blocks_[blockid];

  DEBUG_ASSERT(!block->is_owner);
  ReadTransaction read_transaction;
  read_transaction.Doit(channel_num_, block->value,
      blockid, 0, n_block_bytes_, block->data);
}

void DistributedCache::MarkDirty_(BlockMetadata *block,
    BlockDevice::blockid_t begin, BlockDevice::blockid_t end) {
}

void DistributedCache::MarkDirty_(BlockMetadata *block) {
  FreeDirtyList_(block);

  block->dirty_ranges = FULLY_DIRTY;
}

void DistributedCache::WritebackDirtyLocal_(BlockDevice::blockid_t blockid) {
  BlockMetadata *block = &blocks_[blockid];

  DEBUG_ASSERT(block->is_owner);
  DEBUG_ASSERT(block->dirty_ranges != NOT_DIRTY);

  local_device_

  FreeDirtyList_(block);
  block->dirty_ranges = NOT_DIRTY;
}

void DistributedCache::FreeDirtyList_(BlockMetadata *block) {
  int rangeid = block->dirty_ranges;
  while (rangeid >= 0) {
    RangeLink *range = &ranges_[range];
    rangeid = range->next;
    range->next = free_range_;
    free_range_ = rangeid;
  }
}

void DistributedCache::WritebackDirtyRemote_(BlockDevice::blockid_t blockid) {
  BlockMetadata *block = &blocks_[blockid];
  int rangeid = block->dirty_ranges;

  DEBUG_ASSERT(!block->is_owner);
  DEBUG_ASSERT(block->diry_ranges != NOT_DIRTY);

  if (rangeid == FULLY_DIRTY) {
    // The entire block is dirty
    WriteTransaction write_transaction;
    write_transaction.Doit(channel_num, block->value, blockid,
        0, n_block_bytes_, block->data);
  } else {
    // Flush each independent dirty range
    while (rangeid >= 0) {
      RangeLink *range = &ranges_[range];
      rangeid = range->next;
      range->next = free_range_;
      free_range_ = rangeid;

      WriteTransaction write_transaction;
      write_transaction.Doit(channel_num, block->value, blockid,
          range->begin, range->end, block->data + range->begin);
    }
  }

  block->dirty_ranges = NOT_DIRTY;
}


//-------------------------------------------------------------------------
//-- PROTOCOL MESSAGES ----------------------------------------------------
//-------------------------------------------------------------------------

/** Net-transferable request operation */
struct DCRequest {
 public:
  enum { QUERY, READ, WRITE, ALLOC } type;
  BlockDevice::blockid_t blockid;
  BlockDevice::offset_t begin;
  BlockDevice::offset_t end;
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
  mem::Copy(request->data(), response->data(), end - begin);
  Send(message);
  Done();
  // no wait necessary
}

void DistributedCache::WriteTransaction::HandleMessage(Message *message) {
  FATAL("No response to DistributedCache::WriteTransaction expected");
}

//-------------------------------------------------------------------------

struct DCQueryResponse {
  ArrayList<DistributedCache::BlockStatus> statuses;
};

void DistributedCache::QueryTransaction::Doit(DistributedCache *cache,
    int channel_num, int peer, BlockDevice::blockid_t blockid,
    BlockDevice::offset_t begin, BlockDevice::offset_t end,
    const char *buffer) {
  Transaction::Init(channel_num);
  Message *message = CreateMessage(peer, sizeof(DCRequest));
  DCRequest *request = reinterpret_cast<DCRequest*>(message->data());
  request->type = DCRequest::QUERY;
  request->blockid = 0;
  request->begin = 0;
  request->end = 0;
  Send(message);
  cond.Wait();
  DCQueryResponse *r = ot::PointerThaw<DCQueryResponse>(
      response->data());
  int my_rank = rpc::rank();
  for (index_t i = 0; i < r->statuses.size(); i++) {
    cache_->HandleStatusInformation_(i, r->statuses[i]);
  }
  response;
  delete response;
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
    case DCRequest::QUERY: {
      DCQueryResponse query_response;
      query_response.statuses.Init(cache_->n_blocks());
      for (index_t i = 0; i < query_response.statuses.size(); i++) {
        query_response.statuses[i].owner = cache_->blocks_[i].owner(cache_);
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
      cache_->Write(block, begin, end,
          static_cast<DCWriteRequest*>(request)->data());
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

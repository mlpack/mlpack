#include "netcache.h"

void RemoteBlockDeviceBackend::Init(BlockDevice *device) {
  blockdev_ = device;
  stats_.Init();
}

void RemoteBlockDeviceBackend::HandleRequest(
    const BlockRequest& request, BlockResponse *response) {
  response->n_block_bytes = blockdev_->n_block_bytes();

  if (request.operation == BlockRequest::WRITE) {
    response->payload.Init();
    blockdev_->Write(request.blockid, request.begin, request.end,
        request.payload.begin());
    response->blockid = request.blockid;

    stats_.RecordWrite(request.end - request.begin);
  } else if (request.operation == BlockRequest::READ) {
    response->payload.Init(request.end - request.begin);
    blockdev_->Read(request.blockid, request.begin, request.end,
        response->payload.begin());
    response->blockid = request.blockid;

    stats_.RecordRead(request.end - request.begin);
  } else if (request.operation == BlockRequest::ALLOC) {
    response->payload.Init();
    response->blockid = blockdev_->AllocBlocks(request.blockid);
  } else if (request.operation == BlockRequest::INFO) {
    response->payload.Init();
    response->blockid = blockdev_->n_blocks();
  } else {
    FATAL("Unknown block operation %d.", request.operation);
  }
}

//-------------------------------------------------------------------------

void HashedRemoteBlockDevice::Init(int channel_in,
    int my_rank_in, int n_machines_in) {
  channel_ = channel_in;
  my_rank_ = my_rank_in;
  n_machines_ = n_machines_in;
  n_block_bytes_ = BIG_BAD_NUMBER;
  n_blocks_ = BIG_BAD_NUMBER;
  stats_.Init();
}

void HashedRemoteBlockDevice::ConnectToMaster() {
  if (my_rank_ != MASTER_RANK) {
    BlockRequest request;

    request.blockid = 0;
    request.begin = 0;
    request.end = 0;
    request.operation = BlockRequest::INFO;
    request.payload.Init();

    Rpc<BlockResponse> response(channel_, MASTER_RANK, request);
    n_blocks_ = response->blockid;
    n_block_bytes_ = response->n_block_bytes;
  }
}

void HashedRemoteBlockDevice::SetLocalDevice(BlockDevice *device) {
  local_device_ = device;

  if (my_rank_ != MASTER_RANK) {
    DEBUG_ASSERT(n_block_bytes_ == local_device_->n_block_bytes());
  } else {
    n_blocks_ = 0;
    n_block_bytes_ = local_device_->n_block_bytes();
  }

  server_.Init(this);
}

void HashedRemoteBlockDevice::Read(blockid_t blockid,
    offset_t begin, offset_t end, char *data) {
  int dest = RankHash_(blockid);

  if (dest == my_rank_) {
    local_device_->Read(LocalBlockId_(blockid), begin, end, data);
  } else {
    BlockRequest request;

    stats_.RecordRead(end - begin);

    request.blockid = blockid;
    request.begin = begin;
    request.end = end;
    request.operation = BlockRequest::READ;
    request.payload.Init();

    Rpc<BlockResponse> response(channel_, dest, request);
    memcpy(data, response->payload.begin(), response->payload.size());
    DEBUG_SAME_INT(size_t(response->payload.size()), size_t(end - begin));
  }
}

void HashedRemoteBlockDevice::Write(blockid_t blockid,
    offset_t begin, offset_t end, const char *data) {
  int dest = RankHash_(blockid);

  if (dest == my_rank_) {
    blockid_t local_blockid = LocalBlockId_(blockid);
    if (local_blockid >= local_device_->n_blocks()) {
      local_device_->AllocBlocks(local_blockid + 1 - local_device_->n_blocks());
    }
    DEBUG_ASSERT(local_blockid < local_device_->n_blocks());
    local_device_->Write(local_blockid, begin, end, data);
  } else {
    BlockRequest request;

    stats_.RecordWrite(end - begin);

    request.blockid = blockid;
    request.begin = begin;
    request.end = end;
    request.operation = BlockRequest::WRITE;
    request.payload.Copy(data, end - begin);

    Rpc<BlockResponse> response(channel_, RankHash_(blockid), request);
  }
}

BlockDevice::blockid_t HashedRemoteBlockDevice::AllocBlocks(
    BlockDevice::blockid_t n_blocks_to_alloc) {
  BlockDevice::blockid_t blockid;
  int dest = MASTER_RANK;

  if (dest == my_rank_) {
    blockid = n_blocks_;
  } else {
    BlockRequest request;

    request.blockid = n_blocks_to_alloc;
    request.begin = 0;
    request.end = 0;
    request.operation = BlockRequest::ALLOC;
    request.payload.Init();

    Rpc<BlockResponse> response(channel_, dest, request);
    blockid = response->blockid;
  }

  n_blocks_ = blockid + n_blocks_to_alloc;

  return blockid;
}

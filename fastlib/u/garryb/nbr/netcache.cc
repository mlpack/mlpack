#include "netblock.h"

void RemoteBlockDeviceBackend::Init(BlockDevice *device) {
  blockdev_ = device;
  n_reads_ = 0;
  n_read_bytes_ = 0;
  n_writes_ = 0;
  n_write_bytes_ = 0;
}

void RemoteBlockDeviceBackend::HandleRequest(
    const BlockRequest& request, BlockResponse *response) {
  response->n_block_bytes = blockdev_->n_block_bytes();

  if (request.operation == BlockRequest::WRITE) {
    response->payload.Init();
    blockdev_->Write(request.blockid, request.begin, request.end,
        request.payload.begin());
    response->blockid = request.blockid;
    
    n_writes_++;
    n_write_bytes_ += request.end - request.begin;
  } else if (request.operation == BlockRequest::READ) {
    response->payload.Init(request.end - request.begin);
    blockdev_->Read(request.blockid, request.begin, request.end,
        response->payload.begin());
    response->blockid = request.blockid;

    n_reads_++;
    n_read_bytes_ += request.end - request.begin;
  } else if (request.operation == BlockRequest::ALLOC) {
    response->payload.Init();
    response->blockid = blockdev_->AllocBlock();
  } else if (request.operation == BlockRequest::INFO) {
    response->payload.Init();
    response->blockid = blockdev_->n_blocks();
  } else {
    FATAL("Unknown block operation %d.", request.operation);
  }
}

void RemoteBlockDeviceBackend::Report(datanode *module) {
  fx_format_result(module, "n_reads", "%"L64"u", n_reads_);
  fx_format_result(module, "n_read_bytes", "%"L64"u", n_read_bytes_);
  fx_format_result(module, "read_ratio", "%f",
      1.0 * n_read_bytes_ / blockdev_->n_blocks() / blockdev_->n_block_bytes());
  fx_format_result(module, "n_writes", "%"L64"u", n_writes_);
  fx_format_result(module, "n_write_bytes", "%"L64"u", n_write_bytes_);
  fx_format_result(module, "write_ratio", "%f",
      1.0 * n_write_bytes_ / blockdev_->n_blocks() / blockdev_->n_block_bytes());
  fx_format_result(module, "n_block_bytes", "%"L64"u",
      uint64(blockdev_->n_block_bytes()));
  fx_format_result(module, "size", "%"L64"u",
      uint64(blockdev_->n_blocks()) * blockdev_->n_block_bytes());
}

//-------------------------------------------------------------------------

void HashedRemoteBlockDevice::Init(int channel_in,
    int my_rank_in, int n_machines_in) {
  channel_ = channel_in;
  my_rank_ = my_rank_in;
  n_machines_ = n_machines_in;
  n_block_bytes_ = BIG_BAD_NUMBER;
  n_blocks_ = BIG_BAD_NUMBER;
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
    n_block_bytes_ = response->n_block_bytes;
    n_blocks_ = response->blockid;
  }
}

void HashedRemoteBlockDevice::SetLocalDevice(BlockDevice *device) {
  local_device_ = device;

  if (my_rank_ != MASTER_RANK) {
    DEBUG_ASSERT(n_block_bytes_ == local_device_->n_block_bytes());
  } else {
    n_blocks_ = local_device_->n_blocks();
    n_block_bytes_ = local_device_->n_block_bytes();
  }

  server_.Init(this);
  server_.RemoteObjectInit(channel_);
}

void HashedRemoteBlockDevice::Read(blockid_t blockid,
    offset_t begin, offset_t end, char *data) {
  int dest = RankHash_(blockid);

  n_blocks_ = max(blockid + 1, blockid);

  if (dest == my_rank_) {
    local_device_->Read(LocalBlockId_(blockid), begin, end, data);
  } else {
    BlockRequest request;

    request.blockid = blockid;
    request.begin = begin;
    request.end = end;
    request.operation = BlockRequest::READ;
    request.payload.Init();

    Rpc<BlockResponse> response(channel_, dest, request);
    memcpy(data, response->payload.begin(), response->payload.size());
    DEBUG_SAME_INT(response->payload.size(), end - begin);
  }
}

void HashedRemoteBlockDevice::Write(blockid_t blockid,
    offset_t begin, offset_t end, const char *data) {
  int dest = RankHash_(blockid);

  n_blocks_ = max(blockid + 1, blockid);

  if (dest == my_rank_) {
    local_device_->Write(LocalBlockId_(blockid), begin, end, data);
  } else {
    BlockRequest request;

    request.blockid = blockid;
    request.begin = begin;
    request.end = end;
    request.operation = BlockRequest::WRITE;
    request.payload.Copy(data, end - begin);

    Rpc<BlockResponse> response(channel_, GetOwner_(blockid), request);
  }
}

BlockDevice::blockid_t HashedRemoteBlockDevice::AllocBlock() {
  BlockDevice::blockid_t blockid;
  int dest = MASTER_RANK;

  if (dest == my_rank_) {
    blockid = n_blocks_;
  } else {
    BlockRequest request;

    request.blockid = 0;
    request.begin = 0;
    request.end = 0;
    request.operation = BlockRequest::ALLOC;
    request.payload.Init();

    Rpc<BlockResponse> response(channel_, dest, request);
    blockid = response->blockid;
  }

  DEBUG_ASSERT(n_blocks_ < blockid + 1);
  n_blocks_ = blockid + 1;

  return blockid;
}

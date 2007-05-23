#include "rpc.h"

Mutex global_mpi_lock;

//-------------------------------------------------------------------------

void RemoteObjectServer::Connect(int destination) {
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Send(NULL, 0, MPI_CHAR, destination, TAG_BORN, MPI_COMM_WORLD);
}
void RemoteObjectServer::Disconnect(int destination) {
  MPI_Send(NULL, 0, MPI_CHAR, destination, TAG_DONE, MPI_COMM_WORLD);
}

void RemoteObjectServer::Init() {
  last_tag_ = TAG_FIRST_AVAILABLE - 1;
  channels_.Init();
}

int RemoteObjectServer::NewTag() {
  return ++last_tag_;
}

void RemoteObjectServer::Register(
    int channel, RawRemoteObjectBackend *backend) {
  backend->RemoteObjectInit(channel);
  if (channel >= channels_.size()) {
    channels_.Resize(channel + 1);
  }
  channels_[channel] = backend;
}

void RemoteObjectServer::Loop(int n_workers_total) {
  ArrayList<char> data_recv;
  ArrayList<char> data_send;
  int n_workers_born = 0;
  int n_workers_done = 0;

  data_send.Init();
  data_recv.Init();

  MPI_Barrier(MPI_COMM_WORLD);

  while (n_workers_done != n_workers_total) {
    MPI_Status status;

    MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    int length;
    MPI_Get_count(&status, MPI_CHAR, &length);
    DEBUG_ASSERT(length != MPI_UNDEFINED);
    data_recv.Resize(length);
    MPI_Recv(data_recv.begin(), data_recv.size(), MPI_CHAR,
        MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

    if (status.MPI_TAG == TAG_BORN) {
      n_workers_born++;
    } else if (status.MPI_TAG == TAG_DONE) {
      n_workers_done++;
    } else {
      channels_[status.MPI_TAG]->HandleRequestRaw(
          &data_recv, &data_send);
      MPI_Send(data_send.begin(), data_send.size(), MPI_CHAR,
          status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD);
    }
  }
}

//-------------------------------------------------------------------------

void RemoteBlockDeviceBackend::Init(BlockDevice *device) {
  blockdev_ = device;
}

void RemoteBlockDeviceBackend::HandleRequest(
    const BlockRequest& request, BlockResponse *response) {
  response->n_block_bytes = blockdev_->n_block_bytes();

  if (request.operation == BlockRequest::WRITE) {
    response->payload.Init();
    blockdev_->Write(request.blockid, request.begin, request.end,
        request.payload.begin());
    response->blockid = request.blockid;
  } else if (request.operation == BlockRequest::READ) {
    response->payload.Init(request.end - request.begin);
    blockdev_->Read(request.blockid, request.begin, request.end,
        response->payload.begin());
    response->blockid = request.blockid;
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

//-------------------------------------------------------------------------

void RemoteBlockDevice::Init(int channel_in, int destination_in) {
  stub_.Init(channel_in, destination_in);

  BlockRequest request;

  request.blockid = 0;
  request.begin = 0;
  request.end = 0;
  request.operation = BlockRequest::INFO;
  request.payload.Init();

  stub_.Lock();
  const BlockResponse *response = stub_.Request(request);
  n_block_bytes_ = response->n_block_bytes;
  n_blocks_ = response->blockid;
  stub_.Unlock();
}

void RemoteBlockDevice::Read(blockid_t blockid,
    offset_t begin, offset_t end, char *data) {
  BlockRequest request;

  request.blockid = blockid;
  request.begin = begin;
  request.end = end;
  request.operation = BlockRequest::READ;
  request.payload.Init();

  stub_.Lock();
  const BlockResponse *response = stub_.Request(request);
  memcpy(data, response->payload.begin(), response->payload.size());
  DEBUG_SAME_INT(response->payload.size(), end - begin);
  stub_.Unlock();
}

void RemoteBlockDevice::Write(blockid_t blockid,
    offset_t begin, offset_t end, const char *data) {
  BlockRequest request;

  request.blockid = blockid;
  request.begin = begin;
  request.end = end;
  request.operation = BlockRequest::WRITE;
  request.payload.Copy(data, end - begin);

  stub_.Lock();
  const BlockResponse *response = stub_.Request(request);
  stub_.Unlock();
}

BlockDevice::blockid_t RemoteBlockDevice::AllocBlock() {
  BlockRequest request;
  BlockDevice::blockid_t blockid;

  request.blockid = 0;
  request.begin = 0;
  request.end = 0;
  request.operation = BlockRequest::ALLOC;
  request.payload.Init();

  stub_.Lock();
  const BlockResponse *response = stub_.Request(request);
  blockid = response->blockid;
  stub_.Unlock();

  return blockid;
}

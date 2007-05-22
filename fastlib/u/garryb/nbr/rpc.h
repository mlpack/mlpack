#ifndef NBR_RPC_H
#define NBR_RPC_H

extern Mutex global_mpi_lock;

/**
 * This class is your interface to an object that's somewhere else.
 */
template<class Request, class Response>
class RemoteObjectStub {
  FORBID_COPY(RemoteObjectStub);

 private:
  ArrayList<char> data_;
  int destination_;
  int channel_;
  Mutex mutex_;
#ifdef DEBUG
  bool locked_; // for debug mode
#endif

 public:
  void Init(int channel_in, int destination_in) {
    channel_ = channel_in;
    destination_ = destination_in;
  }

  const Response *Request(const Request& request) {
    DEBUG_ASSERT(locked_ == true);

    global_mpi_lock.Lock();

    data_.Resize(ot::PointerFrozenSize(request));
    ot::PointerFreeze(request, data_.begin());
    MPI_Send(data_.begin(), data_.size(), MPI_CHAR,
        destination_, channel_, MPI_COMM_WORLD);
    MPI_Status status;
    MPI_Probe(MPI_ANY_SOURCE, channel_, MPI_COMM_WORLD, &status);
    data_.Resize(status.MPI_LENGTH);
    MPI_Recv(data_.begin(), data_.size(), MPI_CHAR,
        destination_, channel_, MPI_COMM_WORLD, &status);

    global_mpi_lock.Unlock();

    return ot::PointerThaw<Response>(data_.begin());
  }

  void Lock() {
    mutex_.Lock();
    DEBUG_ONLY(locked_ = true);
  }

  void Unlock() {
    DEBUG_ONLY(locked_ = false);
    mutex_.Unlock();
  }
};

class RawRemoteObjectBackend {
 private:
  int channel_;

 public:
  virtual ~RawRemoteObjectBackend() {}

  void RemoteObjectInit(int channel_in) {
    channel_ = channel_in;
  }

  virtual void HandleRequestRaw(ArrayList<char> *request,
      ArrayList<char> *response);

  int channel() const {
    return channel_;
  }
};

/**
 * This is how you define the network object on the server.
 */
template<class Request, class Response>
class RemoteObjectBackend
    : public RawRemoteObjectBackend {
 public:
  virtual ~RemoteObjectBackend() {}

  virtual void HandleRequestRaw(ArrayList<char> *raw_request,
      ArrayList<char> *raw_response) {
    const Request* real_request = ot::PointerThaw(raw_request->begin());
    Response real_response;
    HandleRequest(*real_request, &real_response);
    raw_response->Resize(ot::PointerFrozenSize(real_response));
    ot::PointerFreeze(real_response, raw_response->begin());
  }

  virtual void HandleRequest(const Request& request, Response *response) = 0;
};

class RemoteObjectServer {
  FORBID_COPY(RemoteObjectServer);

 public:
  ArrayList<RawRemoteObjectBackend*> channels_;
  int last_tag_;

 public:
  const int TAG_BORN = 0;
  const int TAG_DONE = 1;
  const int TAG_FIRST_AVAILABLE = 2;

 public:
  void Init() {
    last_tag_ = TAG_FIRST_AVAILABLE - 1;
  }

  /**
   * Returns a new tag for use.
   *
   * All machines have to use NewTag in an exactly identical way.
   */
  int NewTag() {
    return ++last_tag_;
  }

  void Register(RawRemoteObjectBackend *channel) {
    if (channel->channel() >= channels_.size()) {
      channels_.Resize(channel->channel() + 1);
    }
    channels_[channel->channel()] = channel_;
  }

  void Loop(int n_workers_total) {
    ArrayList<char> data_recv;
    ArrayList<char> data_send;
    int n_workers_born = 0;
    int n_workers_done = 0;

    data_send.Init();
    data_recv.Init();

    while (n_workers_done != n_workers_total) {
      MPI_Status status;

      MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      data_recv.Resize(status.MPI_LENGTH);
      MPI_Recv(data_recv.begin(), data_recv.size(), MPI_CHAR,
          MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

      if (status.MPI_TAG == TAG_BORN) {
        n_workers_born++;
      } else if (status.MPI_TAG == TAG_DONE) {
        n_workers_done++;
      } else {
        channels_[status.MPI_SOURCE]->HandleRawRequest(
            &data_recv, &data_send);
        MPI_Send(data_send.begin(), data_send.size(), MPI_CHAR,
            status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD);
      }
    }
  }
};

/**
 * Protocol request for networked block devices.
 */
struct BlockRequest {
  BlockDevice::block_t blockid;
  BlockDevice::offset_t begin;
  BlockDevice::offset_t end;
  enum { READ, WRITE, ALLOC } operation;
  ArrayList<char> payload;

  OT_DEF(BlockRequest) {
    OT_MY_OBJECT(remember to make sure it compiles);
    OT_MY_OBJECT(blockid);
    OT_MY_OBJECT(begin);
    OT_MY_OBJECT(end);
    OT_MY_OBJECT(operation);
    OT_MY_OBJECT(payload);
  }
};

/**
 * Protocol response for networked block devices.
 */
struct BlockResponse {
  BlockDevice::blockid_t blockid;
  ArrayList<char> payload;

  OT_DEF(BlockResponse) {
    OT_MY_OBJECT(remember to make sure it compiles);
    OT_MY_OBJECT(blockid);
    OT_MY_OBJECT(payload);
  }
};

class BlockDeviceRemoteObjectBackend
    : public RemoteObjectBackend<BlockRequest, BlockResponse> {
 private:
  BlockDevice *blockdev_;

 public:
  void HandleRequest(const BlockRequest& request, BlockResponse *response) {

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
    } else {
      FATAL("Only valid block operations are READ/WRITE.");
    }
  }
};

/**
 * A block device sitting on another computer.
 *
 * Individual instances of this object are not
 * thread safe?
 */
class BlockDeviceRemote
    : public BlockDevice {
 private:
  RemoteObjectStub<BlockRequest, BlockResponse> stub_;

 public:
  void Init(int channel_in, int destination_in) {
    stub_.Init(channel_in, destination_in);
  }

  virtual void Read(blockid_t blockid,
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
    DEBUG_SAME_INT(response->payload.size() == end - begin);
    stub_.Unlock();
  }

  virtual void Write(blockid_t blockid,
      offset_t begin, offset_t end, const char *data) {
    BlockRequest request;

    request.blockid = blockid;
    request.begin = begin;
    request.end = end;
    request.operation = BlockRequest::WRITE;
    request.payload.Copy(data, end - begin);

    stub_.Lock();
    const BlockResponse *response = stub_.Request(request);
    DEBUG_ASSERT(response->payload.size(), 0);
    stub_.Unlock();
  }

  virtual blockid_t AllocBlock() {
    BlockRequest request;
    BlockDevice::blockid_t blockid;

    request.blockid = 0;
    request.begin = 0;
    request.end = 0;
    request.operation = BlockRequest::ALLOC;
    request.payload.Init();

    stub_.Lock();
    const BlockResponse *response = stub_.Request(request);
    DEBUG_ASSERT(response->payload.size(), 0);
    blockid = response->blockid;
    stub_.Unlock();

    return blockid;
  }

  virtual void Close() {}
};

#endif

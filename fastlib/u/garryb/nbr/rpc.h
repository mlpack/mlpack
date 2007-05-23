/**
 * @file rpc.h
 *
 * Remote procedure call support for FASTlib.  Yay!
 */

#ifndef NBR_RPC_H
#define NBR_RPC_H

#ifdef USE_MPI

#include "blockdev.h"

#include "fastlib/fastlib_int.h"

#include <mpi.h>

extern Mutex global_mpi_lock;

/**
 * This class is your interface to an object that's somewhere else.
 */
template<class Request, class Response>
class RemoteObjectStub {
  FORBID_COPY(RemoteObjectStub);

 private:
  ArrayList<char> data_;
  int channel_;
  int destination_;
  Mutex mutex_;
#ifdef DEBUG
  bool locked_; // for debug mode
#endif

 public:
  RemoteObjectStub() {}
  ~RemoteObjectStub() {}
  
  void Init(int channel_in, int destination_in) {
    data_.Init();
    channel_ = channel_in;
    destination_ = destination_in;
    DEBUG_ONLY(locked_ = false);
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
    int length;
    MPI_Get_count(&status, MPI_CHAR, &length);
    data_.Resize(length);
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
      ArrayList<char> *response) = 0;

  int channel() const {
    return channel_;
  }
};

/**
 * This is how you define the network object on the server.
 */
template<typename Request, typename Response>
class RemoteObjectBackend
    : public RawRemoteObjectBackend {
 public:
  virtual ~RemoteObjectBackend() {}

  virtual void HandleRequestRaw(ArrayList<char> *raw_request,
      ArrayList<char> *raw_response) {
    const Request* real_request =
        ot::PointerThaw<Request>(raw_request->begin());
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
  static const int TAG_BORN = 0;
  static const int TAG_DONE = 1;
  static const int TAG_FIRST_AVAILABLE = 0;

 public:
  static void Connect(int destination);
  static void Disconnect(int destination);

 public:
  RemoteObjectServer() {}
  ~RemoteObjectServer() {}
  
  void Init();

  /**
   * Returns a new tag for use.
   *
   * All machines have to use NewTag in an exactly identical way.
   */
  int NewTag();

  void Register(int channel, RawRemoteObjectBackend *backend);

  void Loop(int n_workers_total);
};

/**
 * Protocol request for networked block devices.
 */
struct BlockRequest {
  BlockDevice::blockid_t blockid;
  BlockDevice::offset_t begin;
  BlockDevice::offset_t end;
  enum { READ, WRITE, ALLOC, INFO } operation;
  ArrayList<char> payload;

  OT_DEF(BlockRequest) {
    OT_MY_OBJECT(blockid);
    OT_MY_OBJECT(begin);
    OT_MY_OBJECT(end);
    OT_MY_OBJECT(operation);
    OT_MY_OBJECT(payload);
  }
};

struct DataGetterRequest {
  enum { GET_DATA } operation;
  
  OT_DEF(DataGetterRequest) {
    OT_MY_OBJECT(operation);
  }
};

template<typename T>
struct DataGetterResponse {
  T data;
  
  OT_DEF(DataGetterResponse) {
    OT_MY_OBJECT(data);
  }
};

template<typename T>
class DataGetterBackend
    : public RemoteObjectBackend<DataGetterRequest, DataGetterResponse<T> > {
 private:
  const T* data_;

 public:
  void Init(const T* data_in) {
    data_ = data_in;
  }

  void HandleRequest(
      const DataGetterRequest& request, DataGetterResponse<T> *response) {
    response->data.Copy(*data_);
  }
};

template<typename T>
class RemoteDataGetter {
 private:
  RemoteObjectStub<DataGetterRequest, DataGetterResponse<T> > stub_;
 
 public:
  void Init(int channel, int destination) {
    stub_.Init(channel, destination);
  }
  
  void GetData(T *result) {
    DataGetterRequest request;
    
    request.operation = DataGetterRequest::GET_DATA;
    
    stub_.Lock();
    const DataGetterResponse<T> *response = stub_.Request(request);
    result->Copy(response->data);
    stub_.Unlock();
  }
};

/**
 * Protocol response for networked block devices.
 */
struct BlockResponse {
  unsigned int n_block_bytes;
  BlockDevice::blockid_t blockid;
  ArrayList<char> payload;

  OT_DEF(BlockResponse) {
    OT_MY_OBJECT(n_block_bytes);
    OT_MY_OBJECT(blockid);
    OT_MY_OBJECT(payload);
  }
};

class RemoteBlockDeviceBackend
    : public RemoteObjectBackend<BlockRequest, BlockResponse> {
 private:
  BlockDevice *blockdev_;

 public:
  void Init(BlockDevice *device);
  void HandleRequest(const BlockRequest& request, BlockResponse *response);
};

/**
 * A block device sitting on another computer.
 *
 * Individual instances of this object are not
 * thread safe?
 */
class RemoteBlockDevice
    : public BlockDevice {
 private:
  RemoteObjectStub<BlockRequest, BlockResponse> stub_;

 public:
  void Init(int channel_in, int destination_in);
  virtual void Read(blockid_t blockid,
      offset_t begin, offset_t end, char *data);
  virtual void Write(blockid_t blockid,
      offset_t begin, offset_t end, const char *data);
  virtual blockid_t AllocBlock();
};

#endif

#endif

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
 * A single remote procedure call transaction.
 *
 * This automatically handles all the memory management that is involved
 * with marshalling and unmarshalling, freeing memory when the Rpc object
 * is destructed.
 */
template<class ResponseObject>
class Rpc {
 private:
  char *data_;
  int channel_;
  int destination_;
 
 public:
  template<typename RequestObject>
  Rpc(int channel, int destination, const RequestObject& request) {
    Request(channel, destination, request);
  }
  Rpc() {
  }
  ~Rpc() {
    mem::Free(data_);
  }

  template<typename RequestObject>
  ResponseObject *Request(
      int channel, int destination, const RequestObject& request) {
    int length;
    length = ot::PointerFrozenSize(request);
    data_ = mem::Alloc<char>(length);
    ot::PointerFreeze(request, data_);
    MPI_Send(data_, length, MPI_CHAR, destination_, channel_, MPI_COMM_WORLD);
    MPI_Status status;
    MPI_Probe(MPI_ANY_SOURCE, channel_, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, MPI_CHAR, &length);
    data_ = mem::Resize(data_, length);
    MPI_Recv(data_, length, MPI_CHAR,
        destination_, channel_, MPI_COMM_WORLD, &status);
    return ot::PointerThaw<ResponseObject>(data_);
  }

  operator ResponseObject *() {
    return reinterpret_cast<ResponseObject*>(data_);
  }
  ResponseObject* operator ->() {
    return reinterpret_cast<ResponseObject*>(data_);
  }
  ResponseObject& operator *() {
    return *reinterpret_cast<ResponseObject*>(data_);
  }
  operator const ResponseObject *() const {
    return *reinterpret_cast<ResponseObject*>(data_);
  }
  const ResponseObject* operator ->() const {
    return reinterpret_cast<ResponseObject*>(data_);
  }
  const ResponseObject& operator *() const {
    return *reinterpret_cast<ResponseObject*>(data_);
  }
};

// /**
//  * This class is your interface to an object that's somewhere else.
//  */
// template<class RequestObject, class ResponseObject>
// class RemoteObjectStub {
//   FORBID_COPY(RemoteObjectStub);
// 
//  private:
//   ArrayList<char> data_;
//   int channel_;
//   int destination_;
//   Mutex mutex_;
// #ifdef DEBUG
//   bool locked_; // for debug mode
// #endif
// 
//  public:
//   RemoteObjectStub() {}
//   ~RemoteObjectStub() {}
//   
//   void Init(int channel_in, int destination_in) {
//     data_.Init();
//     channel_ = channel_in;
//     destination_ = destination_in;
//     DEBUG_ONLY(locked_ = false);
//   }
// 
//   const ResponseObject *Request(const RequestObject& request) {
//     DEBUG_ASSERT(locked_ == true);
// 
//     global_mpi_lock.Lock();
// 
//     data_.Resize(ot::PointerFrozenSize(request));
//     ot::PointerFreeze(request, data_.begin());
//     MPI_Send(data_.begin(), data_.size(), MPI_CHAR,
//         destination_, channel_, MPI_COMM_WORLD);
//     MPI_Status status;
//     MPI_Probe(MPI_ANY_SOURCE, channel_, MPI_COMM_WORLD, &status);
//     int length;
//     MPI_Get_count(&status, MPI_CHAR, &length);
//     data_.Resize(length);
//     MPI_Recv(data_.begin(), data_.size(), MPI_CHAR,
//         destination_, channel_, MPI_COMM_WORLD, &status);
// 
//     global_mpi_lock.Unlock();
// 
//     return ot::PointerThaw<ResponseObject>(data_.begin());
//   }
// 
//   void Lock() {
//     mutex_.Lock();
//     DEBUG_ONLY(locked_ = true);
//   }
// 
//   void Unlock() {
//     DEBUG_ONLY(locked_ = false);
//     mutex_.Unlock();
//   }
// };

class RawRemoteObjectBackend {
 private:
  int channel_;

 public:
  virtual ~RawRemoteObjectBackend() {}

  virtual void HandleRequestRaw(ArrayList<char> *request,
      ArrayList<char> *response) = 0;

  void RemoteObjectInit(int channel_in) {
    channel_ = channel_in;
  }

  int channel() const {
    return channel_;
  }
};

/**
 * This is how you define the network object on the server.
 */
template<typename RequestObject, typename ResponseObject>
class RemoteObjectBackend
    : public RawRemoteObjectBackend {
 public:
  virtual ~RemoteObjectBackend() {}

  virtual void HandleRequestRaw(ArrayList<char> *raw_request,
      ArrayList<char> *raw_response) {
    const RequestObject* real_request =
        ot::PointerThaw<RequestObject>(raw_request->begin());
    ResponseObject real_response;
    HandleRequest(*real_request, &real_response);
    raw_response->Resize(ot::PointerFrozenSize(real_response));
    ot::PointerFreeze(real_response, raw_response->begin());
  }

  virtual void HandleRequest(const RequestObject& request, ResponseObject *response) = 0;
};

class RpcServer {
  FORBID_COPY(RpcServer);
  friend class RpcServerTask;

 public:
  ArrayList<RawRemoteObjectBackend*> channels_;
  Thread thread_;
  bool should_stop_;

 public:
  static const int TAG_BORN = 0;
  static const int TAG_DONE = 1;

 public:
  RpcServer() {}
  ~RpcServer() {}
  
  void Init();

  void Register(int channel, RawRemoteObjectBackend *backend);
  void Start();
  void Stop();
  
 private:
  void Loop_();
};

struct DataGetterRequest {
  enum Operation { GET_DATA } operation;
  
  OT_DEF(DataGetterRequest) {
    OT_MY_OBJECT(operation);
  }
};

template<typename T>
class DataGetterBackend
    : public RemoteObjectBackend<DataGetterRequest, T> {
 private:
  const T* data_;

 public:
  void Init(const T* data_in) {
    data_ = data_in;
  }

  void HandleRequest(
      const DataGetterRequest& request, T *response) {
    response->Copy(*data_);
  }
};

template<typename T>
void GetRemoteData(int channel, int destination, T* result) {
  DataGetterRequest request;
  request.operation = DataGetterRequest::GET_DATA;
  Rpc<T> response(channel, destination, request);
  result->Copy(*response);
}

#endif

#endif

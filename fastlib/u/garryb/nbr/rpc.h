/**
 * @file rpc.h
 *
 * Remote procedure call support for FASTlib.  Yay!
 */

#ifndef NBR_RPC_H
#define NBR_RPC_H

#include "blockdev.h"

#include "fastlib/fastlib_int.h"

#include "rpc_sock.h"

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
  ArrayList<char> data_;
  ResponseObject *response_object_;
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
  }

  template<typename RequestObject>
  ResponseObject *Request(
      int channel, int destination, const RequestObject& request) {
    int length;
    data_.Init(ot::PointerFrozenSize(request)
        + RpcImpl::request_header_size);
    ot::PointerFreeze(request, data_.begin() + RpcImpl::request_header_size);
    data_ = RpcImpl::SendReceive(channel_, destination_, &data_);
    response_object_ = ot::PointerThaw<ResponseObject>(
        data_.ptr() + RpcImpl::response_header_size);
    return response_object_;
  }

  operator ResponseObject *() {
    return response_object_;
  }
  ResponseObject* operator ->() {
    return response_object_;
  }
  ResponseObject& operator *() {
    return *response_object_;
  }
  operator const ResponseObject *() const {
    return response_objet_;
  }
  const ResponseObject* operator ->() const {
    return response_object_;
  }
  const ResponseObject& operator *() const {
    return *response_object_;
  }
};

class RawRemoteObjectBackend {
 private:
  int channel_;

 public:
  virtual ~RawRemoteObjectBackend() {}

  virtual void HandleRequestRaw(
      ArrayList<char> *buffer,
      size_t in_header_size, size_t out_header_size);

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

  virtual void HandleRequestRaw(
      ArrayList<char> *buffer,
      size_t in_header_size, size_t out_header_size);
  virtual void HandleRequest(const RequestObject& request,
      ResponseObject *response) = 0;
};

template<typename RequestObject, typename ResponseObject>
void RemoteObjectBackend<RequestObject, ResponseObject>::HandleRequestRaw(
    ArrayList<char> *buffer,
    size_t in_header_size, size_t out_header_size) {
  const RequestObject* real_request =
      ot::PointerThaw<RequestObject>(buffer->begin() + in_header_size);
  ResponseObject real_response;
  HandleRequest(*real_request, &real_response);
  buffer->Resize(out_header_size + ot::PointerFrozenSize(real_response));
  ot::PointerFreeze(real_response, raw_response->begin() + out_header_size);
}


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

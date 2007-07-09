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
  FORBID_COPY(Rpc);
 private:
  struct RpcRequestTransaction : public Transaction {
    FORBID_COPY(RpcRequestTransaction);

   public:
    Message* response;
    Mutex mutex;
    WaitCondition cond;

   public:
    RpcRequestTransaction() {}
    ~RpcRequestTransaction() {}

    Message *Doit(int channel, int peer, const RequestObject& request) {
      Transaction::Init(channel);
      Message *message = CreateMessage(peer, channel,
          ot::PointerFrozenSize(request));
      ot::PointerFreeze(request, message->data());
      mutex.Lock();
      Send(message);
      response = NULL;
      do {
        cond.Wait(&mutex);
      } while (response == NULL);
      mutex.Unlock();
      return response;
    }

    void HandleMessage(Message *message) {
      mutex.Lock();
      response = message;
      mutex.Unlock();
      cond.Signal();
      // TODO: Handle done
      Done();
    }
  };
  
 private:
  Message *response_;
  ResponseObject *response_object_;

 public:
  template<typename RequestObject>
  Rpc(int channel, int peer, const RequestObject& request) {
    Request(channel, peer, request);
  }
  Rpc() {
  }
  ~Rpc() {
    delete response_;
  }

  template<typename RequestObject>
  ResponseObject *Request(
      int channel, int peer, const RequestObject& request) {
    RpcRequestTransaction transaction;
    response_ = transaction.Doit(channel, peer, request);
    response_object_ = ot::PointerThaw<ResponseObject>(response_.data());
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

/**
 * This is how you define the network object on the server.
 */
template<typename RequestObject, typename ResponseObject>
class RemoteObjectBackend : public Channel {
 public:
  // Simple request-response transaction
  class RemoteObjectTransaction() {
    FORBID_COPY(RemoteObjectTransaction);
   private:
    RemoteObjectBackend *inner_;

   public:
    RemoteObjectTransaction(RemoteObjectBackend *inner_in)
     : inner_(inner_in)
     {}

    void HandleMessage(Message *request);
  }

 public:
  virtual ~RemoteObjectBackend() {}

  virtual void HandleRequest(const RequestObject& request,
      ResponseObject *response) = 0;

  RemoteObjectTransaction *GetTransaction(Message *message); {
    return new RemoteObjectTransaction(this);
  }
  
  void Register(int channel_num) {
    RpcImpl::Register(channel_num, this);
  }
};

template<typename RequestObject, typename ResponseObject>
void RemoteObjectBackend<RequestObject, ResponseObject>
    ::RemoteObjectTransaction::HandleMessage(Message *request) {
  const RequestObject* real_request =
      ot::PointerThaw<RequestObject>(request->buffer());
  ResponseObject real_response;
  inner_->HandleRequest(*real_request, &real_response);
  Message *response = CreateMessage(
      request->peer(), ot::PointerFrozenSize(real_response));
  ot::PointerFreeze(real_response, response->buffer());
  Send(response);
  Done();
  delete this;
}

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

namespace rpc {
  template<typename T>
  void GetRemoteData(int channel, int peer, T* result) {
    DataGetterRequest request;
    request.operation = DataGetterRequest::GET_DATA;
    Rpc<T> response(channel, peer, request);
    result->Copy(*response);
  }

  void Barrier(int channel_num);
};

#endif

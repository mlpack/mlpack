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
  template<class RequestObject>
  struct RpcRequestTransaction : public Transaction {
    FORBID_COPY(RpcRequestTransaction);

   public:
    Message* response;
    Mutex mutex;
    WaitCondition cond;

   public:
    RpcRequestTransaction() {}
    virtual ~RpcRequestTransaction() {}

    Message *Doit(int channel, int peer, const RequestObject& request) {
      Transaction::Init(channel);
      Message *message = CreateMessage(peer, ot::PointerFrozenSize(request));
      ot::PointerFreeze(request, message->data());
      response = NULL;
      Send(message);
      mutex.Lock();
      while (response == NULL) {
        cond.Wait(&mutex);
      }
      mutex.Unlock();
      return response;
    }

    void HandleMessage(Message *message) {
      Done();
      mutex.Lock();
      response = message;
      cond.Signal();
      mutex.Unlock();
      // TODO: Handle done
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
    RpcRequestTransaction<RequestObject> transaction;
    response_ = transaction.Doit(channel, peer, request);
    response_object_ = ot::PointerThaw<ResponseObject>(response_->data());
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
    return response_object_;
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
  FORBID_COPY(RemoteObjectBackend);

 public:
  // Simple request-response transaction
  class RemoteObjectTransaction : public Transaction {
    FORBID_COPY(RemoteObjectTransaction);

   private:
    RemoteObjectBackend *inner_;

   public:
    RemoteObjectTransaction() {}
    virtual ~RemoteObjectTransaction() {}
    
    void Init(int channel_num, RemoteObjectBackend *inner_in) { 
      Transaction::Init(channel_num);
      inner_ = inner_in;
    }

    virtual void HandleMessage(Message *request);
    // {
    //  const RequestObject* real_request =
    //      ot::PointerThaw<RequestObject>(request->data());
    //  ResponseObject real_response;
    //  inner_->HandleRequest(*real_request, &real_response);
    //  delete request;
    //  Message *response = CreateMessage(
    //      request->peer(), ot::PointerFrozenSize(real_response));
    //  ot::PointerFreeze(real_response, response->data());
    //  Send(response);
    //  Done();
    //  delete this;
    //}
  };

 public:
  RemoteObjectBackend() {}
  virtual ~RemoteObjectBackend() {}

  virtual void HandleRequest(const RequestObject& request,
      ResponseObject *response) {
    FATAL("Virtuality sucks");
  }

  virtual Transaction *GetTransaction(Message *message) {
    RemoteObjectTransaction *t = new RemoteObjectTransaction();
    t->Init(message->channel(), this);
    return t;
  }

  void Register(int channel_num) {
    rpc::Register(channel_num, this);
  }
};

template<typename RequestObject, typename ResponseObject>
void RemoteObjectBackend<RequestObject, ResponseObject>
    ::RemoteObjectTransaction::HandleMessage(Message *request) {
  const RequestObject* real_request =
      ot::PointerThaw<RequestObject>(request->data());
  ResponseObject real_response;
  inner_->HandleRequest(*real_request, &real_response);
  Message *response = CreateMessage(
      request->peer(), ot::PointerFrozenSize(real_response));
  delete request;
  ot::PointerFreeze(real_response, response->data());
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

  virtual void HandleRequest(const DataGetterRequest& request, T *response);
};

template<typename T>
void DataGetterBackend<T>::HandleRequest(const DataGetterRequest& request,
    T* response) {
  response->Copy(*data_);
}

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

/**
 * @file rpc.h
 *
 * FASTlib's message-passing system.
 *
 * RPC is intended to be in interface for which multiple implmentations can
 * exist.
 */

#ifndef THOR_RPC_H
#define THOR_RPC_H

#include "blockdev.h"
#include "rpc_base.h"
#include "rpc_sock.h"

#include "base/common.h"
#include "col/arraylist.h"

//--------------------------------------------------------------------------

/**
 * A BasicTransaction is meant to be used externally (rather than inherited
 * from), and has the capability to store in it a single message and wait.
 *
 * A BasicTransaction is very useful for one-way or synchronous
 * message-passing.
 */
struct BasicTransaction : public Transaction {
  FORBID_COPY(BasicTransaction);

 private:
  Message *response_;
  DoneCondition cond_;

 public:
  BasicTransaction() {}
  ~BasicTransaction() {
    delete response_;
  }

  void Init(int channel_num) {
    Transaction::Init(channel_num);
    response_ = NULL;
  }

  void WaitDone() {
    cond_.Wait();
  }

  void HandleMessage(Message *message) {
    response_ = message;
    Done();
    cond_.Done();
  }
  
 public:
  Message *response() const {
    return response_;
  }
};

/**
 * A single remote procedure call.
 *
 * This automatically handles all the memory management that is involved
 * with marshalling and unmarshalling, freeing memory when the Rpc object
 * is destructed.
 *
 * This implicitly converts to a pointer of the response object type.
 */
template<class ResponseObject>
class Rpc {
  FORBID_COPY(Rpc);

 private:
  BasicTransaction transaction_;
  ResponseObject *response_object_;

 public:
  /** Concenience constructor that performs the request. */
  template<typename RequestObject>
  Rpc(int channel, int peer, const RequestObject& request) {
    Request(channel, peer, request);
  }
  /** Default constructor -- you must call Request later. */
  Rpc() {
  }
  ~Rpc() {
  }

  /**
   * Initializes this by making a request.
   *
   * Returns the response.  This Rpc object will also implicitly cast to
   * a pointer of the response object type.
   */
  template<typename RequestObject>
  ResponseObject *Request(
      int channel, int peer, const RequestObject& request) {
    transaction_.Init(channel);
    Message *request_msg = transaction_.CreateMessage(
        peer, ot::PointerFrozenSize(request));
    ot::PointerFreeze(request, request_msg->data());

    if (request.requires_response()) {
      transaction_.WaitDone();
      response_object_ = ot::PointerThaw<ResponseObject>(
          transaction_.response()->data());
    } else {
      transaction_.Done();
      response_object_ = NULL;
    }

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
  if (real_request->requires_response()) {
    Message *response = CreateMessage(
      request->peer(), ot::PointerFrozenSize(real_response));
    ot::PointerFreeze(real_response, response->data());
    Send(response);
  }
  Done();
  delete request;
  delete this;
}

//--------------------------------------------------------------------------

template<typename TReductor, typename TData>
class ReduceChannel : public Channel {
  FORBID_COPY(ReduceChannel);

 private:
  class ReduceTransaction : public Transaction {
    FORBID_COPY(ReduceTransaction);

   private:
    ArrayList<Message*> received_;
    int n_received_;
    TData *data_;
    const TReductor *reductor_;
    DoneCondition cond_;

   private:
    void CheckStatus_() {
      if (n_received_ == rpc::n_children()) {
        for (index_t i = 0; i < rpc::n_children(); i++) {
          TData *subdata = ot::PointerThaw<TData>(received_[i]->data());
          reductor_->Reduce(*subdata, data_);
          delete received_[i];
        }
        if (!rpc::is_root()) {
          // Send my subtree's results to my parent.
          Message *message_to_send = CreateMessage(rpc::parent(),
              ot::PointerFrozenSize(*data_));
          ot::PointerFreeze(*data_, message_to_send->data());
          Send(message_to_send);
        }
        rpc::Unregister(channel());
        Done();
        cond_.Done();
      }
    }

   public:
    ReduceTransaction() {}
    virtual ~ReduceTransaction() {}

    void Init(int channel_num, const TReductor *reductor_in, TData *data_inout) {
      Transaction::Init(channel_num);
      reductor_ = reductor_in;
      received_.Init(rpc::n_children());
      for (index_t i = 0; i < received_.size(); i++) {
        received_[i] = NULL;
      }
      n_received_ = 0;
      data_ = data_inout;
      CheckStatus_();
    }

    void Wait() {
      cond_.Wait();
    }

    void HandleMessage(Message *message) { 
      index_t i;
      for (i = rpc::n_children(); i--;) {
        if (message->peer() == rpc::child(i)) {
          break;
        }
      }
      if (unlikely(i < 0)) {
        FATAL("Message from peer #%d unexpected during reduce #%d",
            message->peer(), channel());
      }
      if (received_[i] != NULL) {
        FATAL("Multiple messages from peer #%d during reduce #%d: %p %ld %p %ld %d %d",
            message->peer(), channel(),
            received_[i], long(received_[i]->data_size()),
            message, long(message->data_size()),
            message->channel(), received_[i]->channel());
      }
      received_[i] = message;
      Done(message->peer());

      n_received_++;

      CheckStatus_();
    }
  };

 private:
  ReduceTransaction transaction_;

 public:
  ReduceChannel() {}
  ~ReduceChannel() {}

  void Init(int channel_num, const TReductor *reductor, TData *data) {
    transaction_.Init(channel_num, reductor, data);
    rpc::Register(channel_num, this);
  }
  
  void Wait() {
    transaction_.Wait();
  }

  void Doit(int channel_num, const TReductor& reductor, TData *data) {
    Init(channel_num, &reductor, data);
    Wait();
  }

  Transaction *GetTransaction(Message *message) {
    return &transaction_;
  }
};

//--------------------------------------------------------------------------

template<typename TData>
class Broadcaster : public Channel {
  FORBID_COPY(Broadcaster);

 public:
  typedef TData Data;

 private:
  class BroadcastTransaction : public Transaction {
    FORBID_COPY(BroadcastTransaction);

   public:
    Message *received;
    DoneCondition cond;

   private:
    void SendToChildren_() {
      for (index_t i = 0; i < rpc::n_children(); i++) {
        Message *m = CreateMessage(rpc::child(i), received->data_size());
        mem::CopyBytes(m->data(), received->data(), received->data_size());
        Send(m);
      }
      Done();
      cond.Done();
    }

   public:
    BroadcastTransaction() {}
    virtual ~BroadcastTransaction() { delete received; }

    void SetData(const Data& data) {
      size_t size = ot::PointerFrozenSize(data);
      char *buf = mem::Alloc<char>(size);
      ot::PointerFreeze(data, buf);
      received = new Message();
      received->Init(0, 0, 0, buf, 0, size);
    }

    void DoMaster() {
      SendToChildren_();
    }

    void DoWorker() {
      cond.Wait();
    }

    void HandleMessage(Message *message) {
      DEBUG_ASSERT(message->peer() == rpc::parent());
      received = message;
      SendToChildren_();
    }
  };

 private:
  BroadcastTransaction transaction_;
  Data *data_;

 public:
  Broadcaster() {}
  ~Broadcaster() {}

  void SetData(const Data& data) {
    transaction_.SetData(data);
  }

  void Doit(int channel_num) {
    transaction_.Init(channel_num);
    if (!rpc::is_root()) {
      rpc::Register(channel_num, this);
      transaction_.DoWorker();
      rpc::Unregister(channel_num);
    } else {
      transaction_.DoMaster();
    }
    data_ = ot::PointerThaw<Data>(transaction_.received->data());
  }

  Data &get() const {
    return *data_;
  }

  Transaction *GetTransaction(Message *message) {
    return &transaction_;
  }
};

//--------------------------------------------------------------------------

struct DataGetterRequest {
  enum Operation { GET_DATA } operation;

  bool requires_response() const { return true; }

  OT_DEF_BASIC(DataGetterRequest) {
    OT_MY_OBJECT(operation);
  }
};

/**
 * Server that serves a copiable object to any other machine that requests
 * it v ia GetRmoteData.
 */
template<typename T>
class DataGetterBackend
    : public RemoteObjectBackend<DataGetterRequest, T> {
  FORBID_COPY(DataGetterBackend);

 private:
  T data_;

 public:
  void Init(const T& data_in) {
    ot::Copy(data_in, &data_);
  }
  void Init(const T* data_in) {
    ot::Copy(*data_in, &data_);
  }

  virtual void HandleRequest(const DataGetterRequest& request, T *response);
};

template<typename T>
void DataGetterBackend<T>::HandleRequest(const DataGetterRequest& request,
    T* response) {
  ot::Copy(data_, response);
}

/**
 * An asynchronous message-passing system.
 */
namespace rpc {
  /**
   * Gets data from a peer which has a DataGetterBackend registered
   * on the specified channel.
   */
  template<typename T>
  void GetRemoteData(int channel, int peer, T* result) {
    DataGetterRequest request;
    request.operation = DataGetterRequest::GET_DATA;
    Rpc<T> response(channel, peer, request);
    ot::Copy(*response, result);
  }

  /**
   * Synchronizes all machines at a particular barrier.
   *
   * This is implemented by sending messages up the tree until the root,
   * and sending messages down the tree to all machines.
   */
  void Barrier(int channel_num);

  /**
   * Performs an efficient distributed recution.
   *
   * The value will be the value for all the machines if rpc::is_root().
   * On entry, provide my sole contribution to the recution.
   * If I am not the root, it will contain the value for all direct and
   * indirect children of this node.
   * The operator is assumed to be associative, but not necessarily
   * commutative, and is processed precisely in the order of the computers.
   *
   * The reductor should have a public method:
   *
   * <code>Reduce(const TData& right_hand, TData* left_hand_to_modify) const</code>
   *
   * The right-hand-side is passed first because it is not modified, but the
   * left-hand-side is being modified.
   *
   * @param channel_num a unique channel number associated with this
   * @param reductor the reductor object
   * @param value on entry, my part of the contribution; on output, the
   *        reduced value for the subtree of processes rooted at the current
   */
  template<typename TReductor, typename TData>
  void Reduce(int channel_num, const TReductor& reductor, TData *value) {
    ReduceChannel<TReductor, TData> channel;
    channel.Doit(channel_num, reductor, value);
  }
};

#endif

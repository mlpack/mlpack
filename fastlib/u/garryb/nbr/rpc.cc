#include "rpc.h"

Mutex global_mpi_lock;

//-------------------------------------------------------------------------

/**
 * A transient channel just for the purpose of having barriers.
 */
class BarrierChannel : public Channel {
  FORBID_COPY(BarrierChannel);

 private:
  class BarrierTransaction : public Transaction {
    FORBID_COPY(BarrierTransaction);

   private:
    int n_received_;
    bool done_;
    Mutex mutex_;
    WaitCondition cond_;

   private:
    void DoMessage_(int peer) {
      Message *message = CreateMessage(peer, 0);
      // send a blank message -- this is for synchronization purposes only
      Send(message);
    }

    void CheckState_() {
      if (n_received_ >= rpc::children().size()) {
        if (rpc::is_root() || n_received_ > rpc::children().size()) {
          // Tell the kids that the root is ready
          for (int i = 0; i < rpc::children().size(); i++) {
            //fprintf(stderr, "barrier: Message to %d\n", rpc::children()[i]);
            DoMessage_(rpc::children()[i]);
          }
          Done();
          rpc::Unregister(channel());
          mutex_.Lock();
          done_ = true;
          cond_.Signal();
          mutex_.Unlock();
        } else {
          // Tell parent that all my kids are ready
          //fprintf(stderr, "barrier: Message to parent %d\n", rpc::parent());
          DoMessage_(rpc::parent());
        }
      }
    }

    bool IsValidSender_(int peer) {
      if (n_received_ == rpc::children().size()) {
        return peer == rpc::parent();
      } else {
        for (int i = 0; i < rpc::children().size(); i++) {
          if (peer == rpc::children()[i]) {
            return true;
          }
        }
        return false;
      }
    }

   public:
    BarrierTransaction() {}
    virtual ~BarrierTransaction() {}

    void Doit(int channel_num) {
      Transaction::Init(channel_num);
      n_received_ = 0;
      done_ = false;
      CheckState_();
      mutex_.Lock();
      while (!done_) {
        cond_.Wait(&mutex_);
      }
      mutex_.Unlock();
    }

    void HandleMessage(Message *message) { 
      //fprintf(stderr, "barrier: Message from %d\n", message->peer());
      if (unlikely(!IsValidSender_(message->peer()))) {
        FATAL("Message from %d unexpected during barrier #%d with n_received=%d",
            message->peer(), channel(), n_received_);
      }
      delete message;
      n_received_++;
      CheckState_();
    }
  };

 private:
  BarrierTransaction transaction_;

 public:
  BarrierChannel() {}
  virtual ~BarrierChannel() {}

  void Doit(int channel_num) {
    //fprintf(stderr, "barrier: I exist\n");
    rpc::Register(channel_num, this);
    transaction_.Doit(channel_num);
  }

  Transaction *GetTransaction(Message *message) {
    // TODO: Insert checks to make sure a new barrier isn't beginning
    return &transaction_;
  }
};

void rpc::Barrier(int channel_num) {
  BarrierChannel barrier;
  barrier.Doit(channel_num);
}

//-------------------------------------------------------------------------
/*
class RpcServerTask : public Task {
 private:
  RpcServer *server_;

 public:
  RpcServerTask(RpcServer *server_in) : server_(server_in) {}

  void Run() {
    server_->Loop_();
    delete this;
  }
};

void RpcServer::Start() {
  should_stop_ = false;
  thread_.Init(new RpcServerTask(this));
  thread_.Start();
}
void RpcServer::Stop() {
  should_stop_ = true;
  thread_.WaitStop();
}

void RpcServer::Init() {
  channels_.Init();
}

void RpcServer::Register(int channel, RawRemoteObjectBackend *backend) {
  DEBUG_ASSERT(channel >= 10);
  backend->RemoteObjectInit(channel);
  if (channel >= channels_.size()) {
    index_t oldsize = channels_.size();
    channels_.Resize(channel + 1);
    for (index_t i = oldsize; i < channels_.size(); i++) {
      channels_[i] = NULL;
    }
  }
  channels_[channel] = backend;
}

void RpcServer::Loop_() {
  ArrayList<char> data_recv;
  ArrayList<char> data_send;

  data_send.Init();
  data_recv.Init();

  MPI_Barrier(MPI_COMM_WORLD);

  while (!should_stop_) {
    MPI_Status status;

    MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    int length;
    MPI_Get_count(&status, MPI_CHAR, &length);
    DEBUG_ASSERT(length != MPI_UNDEFINED);
    data_recv.Resize(length);
    MPI_Recv(data_recv.begin(), data_recv.size(), MPI_CHAR,
        status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, &status);

    {
      int channel = status.MPI_TAG;
      while (channel >= channels_.size() || channels_[channel] == NULL) {
        abort();
      }
      channels_[channel]->HandleRequestRaw(
          &data_recv, &data_send);
      MPI_Send(data_send.begin(), data_send.size(), MPI_CHAR,
          status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD);
    }
  }
}
*/

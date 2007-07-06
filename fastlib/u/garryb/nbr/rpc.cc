#include "rpc.h"

Mutex global_mpi_lock;

//-------------------------------------------------------------------------

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

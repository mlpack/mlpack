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
        status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, &status);

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

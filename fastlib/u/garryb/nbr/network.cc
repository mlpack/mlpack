#include "network.h"

void NetworkThread::EnsureBuffer_(size_t size) {
  index_t n_chars = size;
  
  if (unlikely(buffer_.size() < size)) {
    buffer_.Resize(size);
  }
}

void NetworkThread::SendPacket(int channel, int destination,
    size_t header_size, const char *header,
    size_t payload_size, const char *payload) {
  index_t total_size = sizeof(uint64) + header_size + payload_size;
  
  EnsureBuffer_(total_size);
  
  char *pos = buffer_.begin();
  *reinterpret_cast<uint64*>(pos) = channel;
  pos += sizeof(uint64);
  mem::CopyBytes(pos, header, header_size);
  pos += header_size;
  mem::CopyBytes(pos, payload, payload_size);
  pos += payload_size;
  DEBUG_ASSERT(buffer_.begin() + total_size == pos);
  
  MPI_Send(buffer_.begin(), total_size, MPI_CHAR, destination, tag_,
      MPI_COMM_WORLD);
}

void NetworkThread::Run() {
  while (!should_terminate_) {
    MPI_Status status;
    MPI_Recv(buffer_.begin(), buffer_.size(), MPI_CHAR,
        MPI_ANY_SOURCE, tag_ + 1, MPI_COMM_WORLD, &status);
    
    const *pos = buffer_.begin();
    int channel = *reinterpret_cast<uint64*>(pos);
    pos += sizeof(uint64);
    
    channels_[channel]->HandleIncomingRaw(
        status.MPI_SOURCE, size - sizeof(uint64), pos);
  }
}

void NetworkThread::Register(RawChannel *channel) {
  if (channel->channel() >= channels_.size()) {
    channels_.Resize(channel->channel() + 1);
  }
  channels_[channel->channel()] = channel;
  EnsureBuffer(sizeof(uint64) + channel->max_packet_size());
}


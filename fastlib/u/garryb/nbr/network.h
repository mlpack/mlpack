#ifndef NBR_NETWORK_H
#define NBR_NETWORK_H

class RawChannel;

class NetworkThread : public Task {
 private:
  static NetworkThread *instance_;
  
 public:
  static void SendPacket(int channel, int destination,
      size_t header_size, const char *header,
      size_t payload_size, const char *payload) {
    instance_->SendPacket(channel, destination,
        header_size, header, payload_size, payload);
  }
  
  static void Stop() {
    instance_->Stop();
  }
  
  static NetworkThread *instance() {
    return instance_;
  }
  
 private:
  int tag_;
  bool should_terminate_;
  ArrayList<char> buffer_;
  ArrayList<RawChannel*> channels_;
 
 public:
  NetworkThread()
   : tag_(0)
   , should_terminate_(false) {
    buffer_.Init();
    channels_.Init();
  }
  
  void Run();

  void SendPacket(int channel, int destination,
      size_t header_size, const char *header,
      size_t payload_size, const char *payload);
  
  void Stop() {
    should_terminate_ = true;
  }
  
  void Register(RawChannel *channel);
  
 private:
  void EnsureBuffer(size_t size);
};

class RawChannel {
 private:
  int channel_;
  
 private:
  RawChannel(int channel_in)
    : channel_(channel_in)
   {}
  
  int channel() const {
    return channel_;
  }
  
  virtual size_t max_packet_size() const = 0;
  virtual void HandleIncomingRaw(char *packet);
  
  void SendPacketRaw(int destination,
      size_t header_size, const char *header,
      size_t payload_size, const char *payload) {
    NetworkThread::SendPacket(channel(), destination,
        header_size, header,
        payload_size, payload);
  }
};

template<typename TPacketHader>
class Channel {
 public:
  typedef TPacketHeader PacketHeader;
  
 private:
  size_t max_payload_size_;
  
 public:
  Channel(int channel_in, size_t max_payload_size_in)
    : RawChannel(channel_in)
    , max_payload_size_(max_payload_size_in)
   {}
  
  size_t header_size() const {
    return stride_align(sizeof(PacketHeader), long);
  }
  
  size_t max_payload_size() const {
    return max_payload_size_;
  }
  
  size_t max_packet_size() const {
    return header_size() + max_payload_size();
  }
  
  virtual void HandleIncomingRaw(int sender, size_t size, const char *packet);
  
  void SendPacket(int destination, const PacketHeader& header,
      size_t payload_size, const char *payload);
  
 public:
  virtual void HandleIncomingPacket(int sender, const PacketHeader& header,
      size_t payload_size, const char *payload) = 0;
};

template<typename TPacketHader>
void Channel<TPacketHeader>::HandleIncomingRaw(
    int sender, size_t size, const char *packet) {
  DEBUG_ASSERT(size >= header_size());
  const PacketHeader *header = reinterpret_cast<const PacketHeader*>(packet);
  const char *payload = header + header_size();
  HandleIncomingPacket(sender, *header, size - header_size(), payload);
}

template<typename TPacketHader>
void Channel<TPacketHeader>::SendPacket(int destination,
    const PacketHeader& header, size_t payload_size, const char *payload) {
  SendPacketRaw(destination,
      header_size(), reinterpret_cast<const char *>(&header_),
      payload_size, packet);
}

#endif


struct BlockAddress {
  OT_DEF(BlockAddress) {
    OT_MY_OBJECT(block);
    OT_MY_OBJECT(offset_begin);
    OT_MY_OBJECT(offset_end);
  }
  
 public:
  BlockDevice::blockid_t block;
  BlockDevice::offset_t offset_begin;
  BlockDevice::offset_t offset_end;
  
 public:
  BlockDevice::offset_t payload_size() const {
    return offset_end - offset_beign;
  }
};


struct BlockPacket {
  OT_DEF(BlockRequest) {
    OT_MY_OBJECT(transaction_id);
    OT_MY_OBJECT(operation);
    OT_MY_OBJECT(address);
  }
  
 public:
  uint32 transaction_id;
  enum { READ, WRITE } operation;
  BlockAddress address;
};

class BlockDeviceChannel
    : Channel<BlockPacket> {
 private:
  BlockDevice *device_;
  
 public:
  BlockDeviceChannel(int channel_in, BlockDevice *device_in)
    : Channel(channel_in, device_in->n_block_bytes())
    , device_(device_in) {
  }

  virtual void HandleIncomingPacket(
      int sender, const PacketHeader& header,
      size_t payload_size, const char *payload) {
    if (header.operation == BlockPacket::READ) {
      Send(channel(), source, );
    } else {
      
    }
  }
};

 private:
  ArrayList<BlockDevice*> blockdevs_;
  
 public:
  


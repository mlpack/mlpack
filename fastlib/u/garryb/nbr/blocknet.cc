
struct BlockLocation {
  OT_DEF(BlockLocation) {
    OT_MY_OBJECT(blockid);
    OT_MY_OBJECT(offset_begin);
    OT_MY_OBJECT(offset_end);
  }

 public:
  BlockDevice::blockid_t blockid;
  BlockDevice::offset_t offset_begin;
  BlockDevice::offset_t offset_end;

 public:
  BlockDevice::offset_t payload_size() const {
    return offset_end - offset_beign;
  }
};


struct BlockHeader {
  OT_DEF(BlockRequest) {
    OT_MY_OBJECT(transaction_id);
    OT_MY_OBJECT(operation);
    OT_MY_OBJECT(location);
    OT_MY_OBJECT(memory_offset);
  }

 public:
  uint32 transaction_id;
  enum { READ, WRITE } operation;
  BlockLocation location;
  char *memory_offset;
};

class BlockDeviceChannel
    : Channel<BlockHeader> {
 private:
  BlockDevice *device_;

 public:
  BlockDeviceChannel(int channel_in, BlockDevice *device_in)
    : Channel(channel_in, device_in->n_block_bytes())
    , device_(device_in) {
  }

  void RequestBlock() {
  }

  virtual void HandleIncomingPacket(
      int sender, const PacketHeader& header,
      size_t payload_size, const char *payload) {
    if (header.operation == BlockHeader::READ) {
      BlockHeader response_header;
      response_header.transaction_id = header.transaction_id;
      response_header.operation = WRITE;
      response_header.location = header.location;
      
      Send(source, response_header, , payload);
    } else {
      (response_header.location.blockid, );
    }
  }
};




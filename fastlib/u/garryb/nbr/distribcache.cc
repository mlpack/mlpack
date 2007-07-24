
class DistributedCache {
  /** Net-transferable request operation */
  struct Request {
   public:
    enum { READ, WRITE, ALLOC } type;
    BlockDevice::blockid_t blockid;
    BlockDevice::offset_t begin;
    BlockDevice::offset_t end;
  };

  /** Net-transferable write operation */
  struct WriteRequest : public Request {
    long long_data[1];

    char *data() {
      return reinterpret_cast<char*>(long_data);
    }

    static size_t size(size_t data_size) {
      return sizeof(WriteRequest) + data_size - sizeof(long_data);
    }
  };

  /** Responds to all messages */
  struct CacheResponseTransaction : public Transaction {
   public:
    DistributedCache* cache_;

   public:
    void Init(DistributedCache *cache_in) {
      cache_ = cache_in;
    }

    void HandleMessage(Message *message) {
      Request *request = reinterpret_cast<Request*>(message->data());
      Message *response;

      switch (request->type) {
       case Request::READ: {
        response = CreateMessage(
            message->peer(), request->end - request->begin);
        cache_->Read(request->blockid, request->begin, request->end,
            response->data());
        Send(response);
        break;
       case Request::WRITE:
        cache_->Write(block, begin, end,
            static_cast<WriteRequest*>(request)->data());
        break;
       default:
        FATAL("Unknown DistributedCache message: %d", int(request->type));
      }

      Done();

      delete message;
      delete this;
    }
  };

  struct CacheChannel : public Channel {
    DistributedCache *cache;
    Transaction *GetTransaction(Message *message) {
      CacheTransaction *t = new CacheTransaction();
      t->Init(cache_);
      return t;
    }
  };

  struct ReadTransaction {
   public:
    Message *response;

   public:
    ~ReadTransaction() {}

    DoneCondition cond;

    void Doit(int channel_num, int peer, BlockDevice::blockid_t blockid,
        BlockDevice::offset_t begin, BlockDevice::offset_t end,
        char *buffer) {
      Transaction::Init(channel_num);
      Message *message = CreateMessage(peer, sizeof(Request));
      Request *request = reinterpret_cast<Request*>(message->data());
      request->type = Request::READ;
      request->blockid = blockid;
      request->begin = begin;
      request->end = end;
      response = NULL;
      Send(message);
      cond.Wait();
      mem::Copy(buffer, response->data(), end - begin);
      delete response;
    }

    void HandleMessage(Message *message) {
      response = message;
      cond.Done();
      Done();
    }
  };

  struct WriteTransaction {
   public:
    void Doit(int channel_num, int peer, BlockDevice::blockid_t blockid,
        BlockDevice::offset_t begin, BlockDevice::offset_t end,
        const char *buffer) {
      Transaction::Init(channel_num);
      Message *message = CreateMessage(peer, WriteRequst::size(end - begin));
      WriteRequest *request = reinterpret_cast<Request*>(message->data());
      request->type = Request::WRITE;
      request->blockid = blockid;
      request->begin = begin;
      request->end = end;
      mem::Copy(request->data(), response->data(), end - begin);
      Send(message);
      Done();
      // no wait necessary
    }
    void HandleMessage(Message *message) {
      delete message;
    }
  };

  struct BlockMetadata {
    char *pointer;
    uint8 owner;
    uint8 is_dirty;
    // shows any write ranges
    uint8 written_ranges = -1;
    /**
     * Local block mapping.
     * If positive, it has a local block mapping.
     */
    BlockDevice::blockid_t local_block;
  };
  /**
   * External link list of write ranges -- condensed
   * during commits.
   */
  struct RangeLink {
    BlockDevice::offset_t begin;
    BlockDevice::offset_t end;
    uint8 next;
  };

 public:
  /* local structures */
  ArrayList<RangeLink> ranges_;
  uint8 free_range_ = -1;
  BlockHandler *handler_;
  // (No overflow device ye)

  /* remote stuff */
  int channel_num_;
  Channel channel_;

 public:
  void MarkWriteRange(BlockDevice::blockid_t begin_block,
      BlockDevice::offset_t begin_offset,
      BlockDevice::blockid_t last_block,
      BlockDevice::offset_t end_offset
      )
  void Commit();
  void Invalidate();
};

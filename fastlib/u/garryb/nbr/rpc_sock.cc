
/*
tasks to complete that must work
 - integrate with rest of code
 - barriers - birth and death
 /- name resolution
   /- REQUIRE IP ADDRESS - Duhh! :-)
 - ssh-bootstrapping
   - start with manual bootstrapping

near future
 - abstract reduce operation that uses the tree structure

things to think about
 - how to shut down cleanly and respond to signals
   - ssh within the c code or ssh externally?
     - try within c code
   - failures
     - use ssh to detect return code
     - if ssh fails, emit an error message and kill yourself
 - fastexec integration
 - accept() - send me your rank? or just ignore rank altogether?

in the far future, abstract out:
 - topology with scatter/gather/reduce mechanics
 - fault tolerance

resolve address - getaddrinfo
register port numbers
*/

RpcSockImpl RpcSockImpl::instance;

RpcSockImpl::Peer::~Peer() {
  if (out_fd >= 0) {
    (void) close(out_fd);
  }
  if (in_fd >= 0) {
    (void) close(out_fd);
  }
}

void RpcSockImpl::Init_() {
  module_ = fx_submodule(fx_root, "rpc", "rpc");
  rank_ = fx_param_int(module_, "rank", "0");
  n_peers_ = fx_param_int_req(module_, "n");
  port_ = fax_param_int(module_, "port", 31415);
  barrier_id = -1;
  barrier_registrants = -1;

  CreatePeers_();
  CalcChildren_();
  Listen_();
  StartPollingThread_();
  Barrier_();
}

void RpcSockImpl::CreatePeers_() {
  TextLineReader reader;

  peers_.Init(n_peers_);
  reader.Open(fx_param_str_req(module_, "peers"));
  
  for (index_t i = 0; i < peers_.size(); i++) {
    Peer *peer = &peers_[i];

    peer->out_fd = -1;
    peer->in_fd = -1;
    peer->in_status = IDLE;
    peer->in_buffer.Init();
    peer->in_buffer_pos = 0;

    // parse the IP address
    mem::DebugPoison(&peer->in_header);
    mem::Zero(&peer->out_addr);
    peer->out_addr.sin_port = htons(port_); // port is globally configured
    if (inet_pton(AF_INET, reader.Peek().c_str(), &peer->out_addr.sin_addr) < 0) {
      FATAL("Invalid IP address [%s] -- must be 1.2.3.4 format\n",
          reader.Peek().c_str());
    }
    reader.Gobble();
  }
}

void RpcSockImpl::CalcChildren_() {
  int m = 1 +
      min(unsigned(n_peers_ - rank_ - 1), unsigned((~rank_) & (rank_-1)));

  for (int i = 1; i < m; i *= 2) {
    *children_.AddBack() = rank_ + m;
  }

  parent_ = rank_ - ((~rank_) & (rank_-1)) - 1;
}

void RpcSockImpl::Listen_() {
  struct sockaddr_in my_address;

  listen_fd_ = socket(AF_INET, SOCK_STREAM, PF_INET); // last param 0?
  mem::Zero(&my_address);
  my_address.sin_family = AF_INET;
  my_address.sin_port = htons(port_);
  my_address.sin_addr.s_addr = htonl(INADDR_ANY);

  if (0 > bind(listen_fd_, (struct sockaddr*)&my_address, sizeof(my_address))) {
    FATAL("Could not bind to selected port %d", port_);
  }

  if (0 > listen(listen_fd_, 10)) {
    FATAL("listen() failed, port %d", port_);
  }
}

void RpcSockImpl::StartPollingThread_() {
  should_stop_ = false;
  polling_task_.Init(this);
  polling_thread_.Init(&polling_task_);
  polling_thread_.Start();
}

void RpcSockImpl::InfectChildren_() {
  open machine file
  for (int i = children_.size(); --i;) {
    int child_rank = children_[i];
    const char *machinename = machines[i];
    create ssh string to execute
    fork process
    if ssh fails, send a signal to the main process
  }
}

void RpcSockImpl::Cleanup_() {
  should_stop_ = true;
  polling_thread_.WaitStop();
  close(listen_fd_);
  peers_.Resize(0); // automatically calls their destructors
}

void RpcSockImpl::Barrier_(int id) {
  AckBarrier_(id);
  PropBarrierDown_();
}

void RpcSockImpl::AckBarrier_(int id) {
  if (barrier_registrants_ == -1) {
    barrier_registrants_ = 1;
    barrier_id_ = id;
  } else if (barrier_id_ != id) {
    FATAL("Barrier mismatch: %d != %d", id, barrier_id_);
  } else {
    ++barrier_registrants_;
  }
  
  if (barrier_registrants_ == 1 + children_.size()) {
    BarrierReady_();
  }
}

void RpcSockImpl::BarrierReady_() {
  if (parent_ != rank_) {
    Header header;

    DEBUG_ASSERT(barrier_registrants_ == 1 + children_.size());
    header.magic = MAGIC;
    header.channel = CHANNEL_BARRIER;
    header.payload_size = 0;
    header.extra = barrier_id_;

    BlockingWrite_(peers_[parent_].out_fd, );

    barrier_registrants_ = -1;
  }
}


class Channel {
 public:
  virtual Transaction *CreateTransaction(Message *message) = 0;
};

class Message {
 private:
  index_t peer_;
  index_t channel_;
  index_t id_;
  char *contents_;

 public:
  void Send();

  char *data() const {
    DEBUG_ASSERT(sizeof(SockConnection::Header) % 16 == 0);
    return contents_ + sizeof(SockConnection::Header);
  }
};

void Message::Send() {
  enqueue the message
}

class Transaction {
 private:
  SockConnection *connection_;

 protected:
  Message *CreateMessage(int destination, size_t size);

 public:
  virtual void HandleMessage(Message *message) = 0;
};

class SockConnection {
  FORBID_COPY(SockConnection);

 private:
  struct Header {
    int32 magic;
    int32 channel;
    int32 transaction_id;
    int32 payload_size;
  };

 private:
  Mutex mutex_;
  int fd_;

  char *read_buffer_;
  size_t read_buffer_pos_;
  size_t read_buffer_size_;
  Header read_header_;

  char *write_buffer_;
  size_t write_buffer_pos_;
  size_t write_buffer_size_;

  ArrayList<Transaction*> incoming_transactions_;
  ArrayList<Transaction*> outgoing_transactions_;

 public:
  SockConnection() {}
  ~SockConnection();

  void Init(int fd);
};

~SockConnection() {
  // TODO: There are more socket functions I might have to call
  (void) close(fd_);
}

void SockConnection::Init(int fd) {
  fd_ = fd;
}

void RpcSockImpl::TryWrite() {
  DEBUG_ASSERT(status_ == WRITING);
  ssize_t bytes_written = write(fd,
      buffer_.begin() + buffer_pos_, buffer_.size() - buffer_pos_);
  if (bytes_written < 0) {
    FATAL("Error writing");
  }
  buffer_pos_ += bytes_written;
  if (buffer_pos_ == buffer_.size()) {
    status_ = IDLE;
    buffer_.Resize(0);
  }
}

bool SockConnection::TryRead() {
  if (!read_buffer_) {
    ssize_t header_bytes = read(fd,
        reinterpret_cast<char*>(&read_header_), sizeof(Header));
    if (header_bytes != sizeof(Header)) {
      FATAL("Error reading packet header: only %d bytes",
          int(header_bytes));
    }
    DEBUG_ASSERT(read_header_.magic == MAGIC);
    read_buffer_ = mem::Alloc(header_.payload_size);
    read_buffer_pos_ = 0;
  }

  DEBUG_ASSERT(status_ == READING);
  ssize_t bytes_read = read(fd,
      read_buffer_ + read_buffer_pos_, read_buffer_size_ - buffer_pos_);

  if (bytes_read < 0) {
    FATAL("Error reading");
  }

  buffer_pos_ += bytes_read;

  if (buffer_pos_ == buffer_.size()) {
    // how do we handle incoming messages?
    Message message;
    message.Init();
    read_buffer_pos_ = 0;
    return true;
  } else {
    return false;
  }
}

    int channel = peer->in_header.channel;
    if (channel == CHANNEL_BARRIER) {
      BarrierAck_(peer->in_header.extra);
    } else {
      if (channel >= channels_.size()) {
        FATAL("Unknown channel %d -- maybe we want to block on channels",
            header.channel);
      }
      RawRemoteObjectBackend *backend = channels_[header.channel];
      if (backend == NULL) {
        FATAL("Unknown channel %d -- maybe we want to block on channels",
            header.channel);
      }
      backend->HandleRequestRaw(&peer->in_buffer, 0, sizeof(Header));
      Header *header = reinterpret_cast<Header*>(peer->in_buffer.begin());
      header->magic = MAGIC;
      header->channel = peer->in_header.channel;
      header->payload_size = peer->in_buffer.size() - sizeof(Header);
      peer->status = Peer::WRITING;
      HandleWrite_(peer);
    }


void RpcSockImpl::PollingLoop_() {
  fd_set read_fds;
  fd_set write_fds;
  fd_set error_fds;

  while (!should_stop_) {
    int maxfd = 0;

    // identify file descriptors we are listening for
    FD_ZERO(&read_fds);
    FD_ZERO(&write_fds);
    FD_ZERO(&error_fds);
    FD_SET(&read_fds, listen_fd_);
    for (index_t i = 0; i < connections_.size(); i++) {
      SockConnection *connection = connections_[i].fd;
      if (connection->fd >= 0) {
        if (connection->status == Peer::WRITING) {
          FD_SET(&write_fds, connection->fd);
        } else {
          FD_SET(&read_fds, connection->fd);
        }
        FD_SET(&error_fds, connection->fd);
        maxfd = max(connection->fd, maxfd);
      }
    }

    // Use a one-second timeout so we can poll for should_stop_ to allow
    // graceful shutdown.
    struct timeval tv;
    tv.tv_sec = 1;
    tv.tv_usec = 0;
    int r = select(maxfd + 1, &read_fds, &write_fds, &error_fds, &ts);

    if (r < 0) {
      NONFATAL("Select failed");
      continue;
    }
    if (r == 0) {
      /* timed out */
      continue;
    }

    if (FD_ISSET(&read_fds, listen_fd_)) {
      sockaddr_in addr;
      socklen_t len = sizeof(addr);
      int new_fd;

      mem::Zero(&addr);
      new_fd = accept(listen_fd_, (struct sockaddr*)&addr, &len);

      if (new_fd >= 0) {
        int i;

        for (i = 0; i < peers_.size(); i++) {
          Peer *peer = &peers_[i];
          if (addr.sin_addr.s_addr == peer->out_addr.sin_addr.s_addr) {
            break;
          }
        }
        if (i == peers_.size()) {
          NONFATAL("Incomming connection from unknown machine");
          (void)close(new_fd);
        } else {
          SockConnection *connection = new SockConnection();
          connection->Init(new_fd);
          *connections_.AddBack() = connection;
          peers_[i].connection = connection;
        }
      }
    }

    for (index_t i = 0; i < peers_.size(); i++) {
      Peer *peer = &peers_[i];
      int fd = peers->in_fd;
      if (fd >= 0) {
        if (FD_ISSET(&error_fds, fd)) {
          // Poor man's way to terminate all processes
          FATAL("Socket error");
        }
        if (FD_ISSET(&write_fds, fd)) {
          HandleWrite_(peer);
        }
        if (FD_ISSET(&read_fds, fd)) {
          HandleRead_(peer);
        }
      }
    }
  }
}


void RpcSockImpl::BlockingWrite_(int fd, size_t len, const char *in_buffer) {
  for (;;) {
    ssize_t written = write(fd, in_buffer, len);

    len -= written;
    if (len == 0) {
      break;
    }
    DEBUG_ASSERT_MSG(written > 0, "error writing");
    in_buffer += written;
  }
}

void RpcSockImpl::BlockingRead_(int fd, size_t len, char *in_buffer) {
  for (;;) {
    if (len == 0) {
      break;
    }
    ssize_t amount_read = read(fd, in_buffer, len);
    len -= amount_read;
    if (amount_read <= 0) {
      assert(amount_read == 0);
      /* end-of-file, fill with zeros */
      memset(in_buffer, 0, len);
      break;
    }
    in_buffer += amount_read;
  }
}

void RpcSockImpl::SendReceive_(int channel, int destination,
    ArrayList<char>* data) {
  Peer *peer = &peers_[destination];
  Header header = reinterpret_cast<Header*>(peer->in_buffer.begin());

  peer->out_mutex.Lock();

  if (peer->out_fd < 0) {
    peer->out_fd = socket(AF_INET, SOCK_STREAM, PF_INET);

    if (0 > connect(peer->out_fd, &peer->out_addr, sizeof(peer->out_addr))) {
      FATAL("connect failed");
    }
  }

  DEBUG_ASSERT_MSG(sizeof(Header) % 16 == 0,
     "Header must be multiple of 16 bytes, or RISC processors won't work due to alignment problems!");

  // Requests must have space preallocated for the header
  header->magic = MAGIC;
  header->channel = channel;
  header->payload_size = data->size();
  BlockingWrite_(peer->out_fd, data->begin(), data->size());

  ssize_t bytes_read = read(peer->out_fd, data->begin(), sizeof(Header));
  if (bytes_read < sizeof(Header)) {
    FATAL("Read error");
  }

  // Responses will just have their header overwritten.
  DEBUG_ASSERT(header->magic == MAGIC);
  DEBUG_ASSERT(header->channel == channel);
  data->Resize(header->payload_size);
  BlockingRead_(fd, header->payload_size, data->begin());

  peer->out_mutex.Unlock();
}

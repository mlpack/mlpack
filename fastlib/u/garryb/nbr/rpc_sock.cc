/**
 * @file rpc_sock.cc
 *
 * Implementation of transaction API using TCP.
 */

/*
tasks to complete that must work
 - integrate with rest of code
   - rpc calls
   - rpc servers
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

//-------------------------------------------------------------------------

void MakeSocketNonBlocking(int fd) {
  fcntl(fd, F_SETFL, O_NONBLOCK);
}

//-------------------------------------------------------------------------

RpcSockImpl RpcSockImpl::instance;

void RpcSockImpl::Init() {
  module_ = fx_submodule(fx_root, "rpc", "rpc");
  rank_ = fx_param_int(module_, "rank", "0");
  n_peers_ = fx_param_int_req(module_, "n");
  port_ = fax_param_int(module_, "port", 31415);
  barrier_id = -1;
  barrier_registrants = -1;
  channels_.Init();
  channels_.default_value() = NULL;

  CreatePeers_();
  CalcChildren_();
  Listen_();
  StartPollingThread_();
}

void RpcSockImpl::Done() {
  should_stop_ = true;
  polling_thread_.WaitStop();
  close(listen_fd_);
  peers_.Resize(0); // automatically calls their destructors
}

void RpcSockImpl::Register(int channel_num, Channel *channel) {
  mutex_.Lock()
  channels_[channel_num] = channel;
  mutex_.Unlock();
  // Inform the polling loop about the new channel so that it can process
  // any events that have been queued
  WakeUpPollingLoop_();
}

void RpcSockImpl::Unregister(int channel_num) {
  mutex_.Lock()
  channels_[channel_num] = NULL;
  mutex_.Unlock();
}

void RpcSockImpl::Send(Message *message) {
  Peer *peer = &peers_[message->peer()];
  peer->mutex.Lock();
  if (peer->outgoing_connection == NULL) {
    peer->outgoing_connection = new SockConnection();
    peer->outgoing_connection->InitConnect(out_addr);
    // Inform polling loop that we have a new socket
    WakeUpPollingLoop_();
  }
  peer->outgoing_connection->Send(message);
  peer->mutex.Unlock();
}

//-- helpers for transaction stuff ----------------------------------------

Message *RpcSockImpl::CreateMessage_(
    int peer, int channel, int transaction_id, size_t size) {
  Message *message = new Message();
  message->Init(peer, channel, transaction_id,
      mem::Alloc<char>(sizeof(SockConnection::Header) + size),
      sizeof(SockConnection::Header), size);
  return message;
}

int RpcSockImpl::DestroyTransaction_(int peer_id, int channel, int id) {
  Peer *peer = &peers_[peer_id];

  mutex_.Lock();
  peer->mutex.Lock();
  if (channel < 0) {
    peer->outgoing_transactions[id] = NULL;
  } else {
    channels_[channel]->CleanupTransaction(incoming_transactions[id]);
    peer->incoming_transactions[id] = NULL;
  }
  peer->mutex.Unlock();
  mutex_.Unlock();

  return id;
}

int RpcSockImpl::AssignTransaction_(int peer_num, Transaction *transaction) {
  Peer *peer = &peers_[peer_num];
  int id;
  mutex.Lock();
  for (id = 0; outgoing_transactions[id] != NULL; id++) {}
  outgoing_transactions[id] = transaction;
  mutex.Unlock();
  return id;
}

//-- helper functions for initialization and the main loop

void RpcSockImpl::CreatePeers_() {
  TextLineReader reader;

  peers_.Init(n_peers_);
  reader.Open(fx_param_str_req(module_, "peers"));

  for (index_t i = 0; i < peers_.size(); i++) {
    Peer *peer = &peers_[i];

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
  int i;

  for (i = 1; i < m; i *= 2) {}
  while (i > 1) {
    i /= 2;
    *children_.AddBack() = rank_ + i;
  }

  parent_ = rank_ - ((~rank_) & (rank_-1)) - 1;
}

void RpcSockImpl::Listen_() {
  int sv[2];
  struct sockaddr_in my_address;

  socketpair(AF_LOCAL, SOCK_STREAM, 0, sv);
  alert_signal_fd_ = sv[0];
  alert_slot_fd_ = sv[1];
  MakeSocketNonblocking(alert_signal_fd_);
  MakeSocketNonblocking(alert_slot_fd_);

  listen_fd_ = socket(AF_INET, SOCK_STREAM, PF_INET); // last param 0?
  MakeSocketNonBlocking(listen_fd_);
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

void RpcSockImpl::WakeUpPollingLoop_() {
  (void) write(alert_signal_fd_, "x", 1);
}

void RpcSockImpl::PollingLoop_() {
  ArrayList<WorkItem> work_items;
  fd_set read_fds;
  fd_set write_fds;
  fd_set error_fds;

  work_items.Init();

  while (!should_stop_) {
    int maxfd;

    FD_ZERO(&read_fds);
    FD_ZERO(&write_fds);
    FD_ZERO(&error_fds);

    FD_SET(&read_fds, listen_fd_);
    FD_SET(&read_fds, alert_slot_fd_);
    maxfd = max(listen_fd_, alert_slot_fd_);

    for (index_t i = 0; i < peers_.size(); i++) {
      Peer *peer = &peers_[i];
      peer->mutex.Lock();
      if (peer->outgoing_connection) {
        peer->outgoing_connection->PrepareSelect(
            &read_fds, &write_fds, &error_fds);
        maxfd = max(maxfd, peer->outgoing_connection->fd());
      }
      if (peer->incoming_connection) {
        peer->incoming_connection->PrepareSelect(
            &read_fds, &write_fds, &error_fds);
        maxfd = max(maxfd, peer->incoming_connection->fd());
      }
      peer->mutex.Unlock();
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

    if (FD_ISSET(&read_fds, alert_signal_fd_)) {
      // Read as much as possible
      char buf[8];
      while (read(alert_signal_fd_, buf, sizeof(buf)) > 0) {}
    }

    if (FD_ISSET(&read_fds, listen_fd_)) {
      sockaddr_in addr;
      socklen_t len = sizeof(addr);
      int new_fd;

      mem::Zero(&addr);
      new_fd = accept(listen_fd_, (struct sockaddr*)&addr, &len);
      MakeSocketNonBlocking(new_fd);

      if (new_fd >= 0) {
        int i;

        for (i = 0; i < peers_.size(); i++) {
          Peer *peer = &peers_[i];
          if (addr.sin_addr.s_addr == peer->addr.sin_addr.s_addr) {
            break;
          }
        }
        if (i == peers_.size()) {
          NONFATAL("Incomming connection from unknown machine");
          (void)close(new_fd);
        } else {
          DEBUG_ASSERT(peers_[i].incoming_connection == NULL);
          SockConnection *connection = new SockConnection();
          connection->Init(new_fd);
          peers_[i].incoming_connection = connection;
        }
      }
    }

    work_items.Resize(0);

    // Gather available messages
    for (index_t i = 0; i < peers_.size(); i++) {
      Peer *peer = &peers_[i];

      // TODO: This loop is probably quite slow.
      // We should look for a faster way of scanning file descriptors --
      // how about a DenseIntMap?
      processable.Init();
      peer->mutex.Lock()
      if (peer->incoming_connection) {
        peer->incoming_connection->HandleSocketEvents(
            &read_fds, &write_fds, &error_fds);
      }
      if (peer->outgoing_connection) {
        peer->outgoing_connection->HandleSocketEvents(
            &read_fds, &write_fds, &error_fds);
      }
      GatherReadyMessages(peer, peer->incoming_connection, &processable);
      peer->mutex.Unlock();
    }

    // Execute work items while we aren't holding any mutexes
    for (index_t i = 0; i < work_items.size(); i++) {
      WorkItem *item = &work_items[i];
      item->transaction->TransactionPreexamineMessage_(item->message);
      // it is transaction's responsibility to delete the message
      item->transaction->HandleMessage(item->message);
    }
  }
}

void RpcSockImpl::GatherReadyMessages_(Peer *peer,
    SockConnection *connection, ArrayList<WorkItem*> *work_items) {
  ArrayList<Message*>* queue = &connection->read_queue();
  int j = 0;

  processable.Init();

  for (index_t i = 0; i < queue->size(); i++) {
    Message *message = (*queue)[i];
    int id = message->transaction_id();
    Transaction *transaction;

    if (message->channel() < 0) {
      transaction = peer->outgoing_transactions[id];
    } else {
      transaction = peer->incoming_transactions[id];
      if (!transaction) {
        // Someone might be registering channels in the background
        mutex_.Lock();
        Channel *channel = channels_[message->channel()];
        if (channel) {
          transaction = channel->GetTransaction(message);
          peer->incoming_transactions[id] = transaction;
        }
        mutex_.Unlock();
      }
    }

    if (transaction != NULL) {
      WorkItem *item = processable.AddBack();
      item->message = message;
      item->transcation = transaction;
    } else {
      (*queue)[j++] = message;
    }
  }
  if (j != i) {
    queue->Resize(j);
  }
}

//-------------------------------------------------------------------------

RpcSockImpl::Peer::Peer() {
  outgoing_connection = NULL;
  incoming_connection = NULL;
  incoming_transactions.Init();
  incoming_transactions.default_value() = NULL;
  outgoing_transactions.Init();
  outgoing_transactions.default_value() = NULL;
}

RpcSockImpl::Peer::~Peer() {
  mutex.Lock();
  if (outgoing_connection) {
    delete outgoing_connection; 
  }
  if (incoming_connection) {
    delete incoming_connection; 
  }
  mutex.Unlock();
}

//-------------------------------------------------------------------------

void Transaction::Init(int channel_in) {
  channel_ = channel_in;
  peers_.Init();
}

Message *Transaction::CreateMessage(int peer, size_t size) {
  Message *message;
  int i;
  int transaction_id;

  for (i = 0; i < peers_.size(); i++) {
    if (peers_[i].peer == peer) {
      break;
    }
  }

  if (i == peers_.size()) {
    peers_.AddBack();
    // TODO: mutex?
    transaction_id = RpcSockImpl::instance.AssignTransaction(peer, this);
    peers_[i].peer = peer;
    peers_[i].channel = channel;
    peers_[i].transaction_id = transaction_id;
  }

  message = RpcSockImpl::CreateMessage_(
      peers_[i].peer, peers_[i].channel, peers_[i].transaction_id, size);

  return message;
}

void Transaction::TransactionPreexamineMessage_(Message *message) {
  for (i = 0; i < peers_.size(); i++) {
    if (peers_[i].peer == peer) {
      break;
    }
  }

  if (i == peers_.size()) {
    peers_.AddBack();
    peers_[i].peer = peer;
    peers_[i].channel = -1;
    peers_[i].transaction_id = message->transaction_id();
  }

  DEBUG_ASSERT(peers_[i].peer == peer);
  DEBUG_ASSERT(peers_[i].transaction_id == message->transaction_id());
}

void Transaction::Send(Message *message) {
  RpcSockImpl::Send(message);
}

void Transaction::Done() {
  for (index_t i = 0; i < peers_.size(); i++) {
    // TODO: Demeter?
    RpcSockImpl::instance.peers_[peers[i].peer].DestroyTransaction(
        peers[i].channel, peers[i].transaction_id);
  }
}

void Transaction::Done(int peer) {
  for (index_t i = 0; i < peers_.size(); i++) {
    // TODO: Demeter?
    if (peer == peers_[i].peer) {
      RpcSockImpl::instance.peers_[peers[i].peer].DestroyTransaction(
          peers[i].channel, peers[i].transaction_id);
      peers_[i] = peers_[peers_.size()-1];
      peers_.PopBack();
    }
  }
}

//-------------------------------------------------------------------------

~SockConnection() {
  // TODO: There are more socket functions I might have to call
  (void) close(fd_);
}

void SockConnection::InitConnect(const struct sockaddr_in &dest) {
  int fd = socket(AF_INET, SOCK_STREAM, PF_INET);

  MakeSocketNonBlocking(fd);

  // make socket non-blocking
  if (0 > connect(fd, &dest, sizeof(struct sockaddr_in))) {
    FATAL("connect failed");
  }

  Init(fd_);
}

void SockConnection::Init(int fd) {
  read_total_ = 0;
  read_message_ = NULL;
  read_buffer_pos_ = 0;
  read_queue_.Init();

  write_total_ = 0;
  write_message_ = NULL;
  write_buffer_pos_ = 0;
  write_queue_.Init();
}

void SockConnection::Send(Message *message) {
  ++write_total;
  Header *header = reinterpret_cast<Header*>(message->buffer());
  header->magic = MAGIC;
  header->channel = message->channel();
  header->transaction_id = message->transaction_id();
  header->data_size = message->data_size();
  if (write_message_ == NULL) {
    write_message_ = message;
  } else {
    write_queue_.Put(write_total_, message);
  }

  wake up network thread
}

void SockConnection::TryWrite() {
  if (write_message_ != NULL) {
    DEBUG_ASSERT(status_ == WRITING);
    ssize_t bytes_written = write(fd,
        write_message_.buffer() + write_buffer_pos_,
        write_message_.buffer_size() - write_buffer_pos_);
    if (bytes_written < 0) {
      if (errno != EAGAIN && errno != EINTR) {
        FATAL("Error writing");
      }
    } else {
      write_buffer_pos_ += bytes_written;
      if (write_buffer_pos_ == write_message_.buffer_size()) {
        delete write_message_;
        if (write_queue_.is_empty()) {
          write_message_ = NULL;
        } else {
          write_message_ = write_queue_.Pop();
        }
      }
    }
  }
}

void SockConnection::TryRead() {
  if (!read_message_) {
    Header header;
    ssize_t header_bytes = read(fd,
        reinterpret_cast<char*>(&header), sizeof(Header));
    if (header_bytes < 0 && (errno == EINTR || errno == EAGAIN)) {
      return;
    }
    if (header_bytes != sizeof(Header)) {
      FATAL("Error reading packet header: read returned %d bytes",
          int(header_bytes));
    }
    DEBUG_ASSERT(header.magic == MAGIC);
    read_message_ = new Message();
    read_message_->Init(peer_, header.channel, header.transaction_id,
        mem::Alloc<char>(header.data_size), 0, header.data_size);
  }
  ssize_t bytes_read = read(fd,
      read_message_.buffer() + read_buffer_pos_,
      read_message_.buffer_size() - read_buffer_pos_);
  if (bytes_read < 0) {
    if (errno != EAGAIN && errno != EINTR) {
      FATAL("Error reading");
    }
  } else {
    read_buffer_pos_ += bytes_read; 
    if (read_buffer_pos_ == read_message_.buffer_size()) {
      ++read_total;
      *read_queue_.AddBack() = message;

      read_message_ = NULL;
      read_buffer_pos_ = 0;
    }
  }
}

void SockConnection::HandleSocketEvents(
    fd_set *read_fds, fd_set *write_fds, fd_set *error_fds) {
  if (FD_ISSET(&error_fds, fd_)) {
    // Poor man's way to terminate all processes
    FATAL("Socket error");
  }
  if (FD_ISSET(&write_fds, fd_)) {
    TryWrite();
  }
  if (FD_ISSET(&read_fds, fd_)) {
    TryRead();
  }
}

void SockConnection::PrepareSelect(
    fd_set *read_fds, fd_set *write_fds, fd_set *error_fds) {
  FD_SET(read_fds, fd_);
  FD_SET(write_fds, fd_);
  FD_SET(error_fds, fd_);
}

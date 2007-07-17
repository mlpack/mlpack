/**
 * @file rpc_sock.cc
 *
 * Implementation of transaction API using TCP.
 */

#include "rpc.h"
#include "rpc_sock.h"

#include "fastlib/fastlib.h"

#include <fcntl.h>
#include <errno.h>
#include <string.h>

//-------------------------------------------------------------------------

/*
tasks to complete that must work
 - smarter transaction lookup so barriers can all share the same number
 - smarter fd_set usage (perhaps a simple linear scan,
     with fd-lookup table?)
 - ssh-bootstrapping
   - start with manual bootstrapping
 /- startup and shutdown
 /- integrate with rest of code
   /- rpc calls
   /- rpc servers
 /- barriers - birth and death
 /- name resolution
   /- REQUIRE IP ADDRESS - Duhh! :-)

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
  fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK);
}

//-------------------------------------------------------------------------

RpcSockImpl RpcSockImpl::instance;

void RpcSockImpl::Init() {
  module_ = fx_submodule(fx_root, "rpc", "rpc");
  n_peers_ = fx_param_int_req(module_, "n");
  rank_ = fx_param_int(module_, "rank", 0);
  DEBUG_ASSERT(rank_ < n_peers_);
  port_ = fx_param_int(module_, "port", 31415);
  channels_.Init();
  channels_.default_value() = NULL;

  CreatePeers_();
  CalcChildren_();
  Listen_();
  StartPollingThread_();

  //fprintf(stderr, "%d: Starting initial barrier\n", rank_);
  rpc::Barrier(0);
  //fprintf(stderr, "%d: Initial barrier over!\n", rank_);
}

void RpcSockImpl::Done() {
  status_ = STOP_SYNC;
  //fprintf(stderr, "%d: Starting final barrier\n", rank_);
  rpc::Barrier(1);
  //fprintf(stderr, "%d: Finished final barrier!\n", rank_);
  status_ = STOP;
  polling_thread_.WaitStop();
  close(listen_fd_);
  peers_.Resize(0); // automatically calls their destructors
}

void RpcSockImpl::Register(int channel_num, Channel *channel) {
  mutex_.Lock();
  //fprintf(stderr, "Registering %d\n", channel_num);
  channels_[channel_num] = channel;
  mutex_.Unlock();
  // Inform the polling loop about the new channel so that it can process
  // any events that have been queued
  WakeUpPollingLoop();
}

void RpcSockImpl::Unregister(int channel_num) {
  mutex_.Lock();
  channels_[channel_num] = NULL;
  mutex_.Unlock();
}

void RpcSockImpl::Send(Message *message) {
  Peer *peer = &peers_[message->peer()];
  peer->mutex.Lock();
  peer->connection.Send(message);
  peer->mutex.Unlock();
}

void RpcSockImpl::WakeUpPollingLoop() {
  (void) write(alert_signal_fd_, "x", 1);
}

void RpcSockImpl::UnregisterTransaction(int peer_id, int channel, int id) {
  Peer *peer = &peers_[peer_id];

  peer->mutex.Lock(); // Lock peer's mutex
  if (channel < 0) {
    peer->incoming_transactions[id] = NULL;
  } else {
    // Old idea -- not really necessary
    //mutex_.Lock(); // Lock mutex -- we are accessing channels
    //channels_[channel]->CleanupTransaction(incoming_transactions[id]);
    //mutex_.Unlock();
    peer->outgoing_transactions[id] = NULL;
  }
  peer->mutex.Unlock();
}

int RpcSockImpl::AssignTransaction(int peer_num, Transaction *transaction) {
  Peer *peer = &peers_[peer_num];
  int id;
  peer->mutex.Lock();
  for (id = 0; peer->outgoing_transactions[id] != NULL; id++) {}
  peer->outgoing_transactions[id] = transaction;
  peer->mutex.Unlock();
  return id;
}

//-- helpers for transaction stuff ----------------------------------------


//-- helper functions for initialization and the main loop

void RpcSockImpl::CreatePeers_() {
  TextLineReader reader;

  peers_.Init(n_peers_);
  reader.Open(fx_param_str_req(module_, "peers"));

  for (index_t i = 0; i < peers_.size(); i++) {
    Peer *peer = &peers_[i];

    peer->connection.Init(i, reader.Peek().c_str(), port_);
    reader.Gobble();
  }
}

void RpcSockImpl::CalcChildren_() {
  int m = 1
      + min(unsigned(n_peers_ - rank_ - 1), unsigned((~rank_) & (rank_-1)));
  int i;

  children_.Init();
  for (i = 1; i < m; i *= 2) {}
  while (i > 1) {
    i /= 2;
    *children_.AddBack() = rank_ + i;
    //fprintf(stderr, "children: %d child: %d\n", rank_, rank_ + i);
  }

  parent_ = rank_ - ((~rank_) & (rank_-1)) - 1;
  //fprintf(stderr, "parent = %d\n", parent_);
}

void RpcSockImpl::Listen_() {
  int sv[2];
  struct sockaddr_in my_address;

  socketpair(AF_LOCAL, SOCK_STREAM, 0, sv);
  alert_signal_fd_ = sv[0];
  alert_slot_fd_ = sv[1];
  MakeSocketNonBlocking(alert_signal_fd_);
  MakeSocketNonBlocking(alert_slot_fd_);

  listen_fd_ = socket(AF_INET, SOCK_STREAM, 0); // last param 0?
  mem::Zero(&my_address);
  my_address.sin_family = AF_INET;
  my_address.sin_port = htons(port_);
  my_address.sin_addr.s_addr = htonl(INADDR_ANY);

  if (0 > bind(listen_fd_, (struct sockaddr*)&my_address, sizeof(my_address))) {
    FATAL("Could not bind to selected port %d on fd %d: %s",
        port_, listen_fd_, strerror(errno));
  }

  if (0 > listen(listen_fd_, 10)) {
    FATAL("listen() failed, port %d", port_);
  }

  MakeSocketNonBlocking(listen_fd_);
}

void RpcSockImpl::StartPollingThread_() {
  status_ = RUN;
  polling_task_.Init(this);
  polling_thread_.Init(&polling_task_);
  polling_thread_.Start();
}

void RpcSockImpl::PollingLoop_() {
  ArrayList<WorkItem> work_items;
  fd_set read_fds;
  fd_set write_fds;
  fd_set error_fds;

  work_items.Init();

  while (status_ != STOP) {
    int maxfd;

    FD_ZERO(&read_fds);
    FD_ZERO(&write_fds);
    FD_ZERO(&error_fds);

    FD_SET(listen_fd_, &read_fds);
    FD_SET(alert_slot_fd_, &read_fds);
    maxfd = max(listen_fd_, alert_slot_fd_);

    for (index_t i = 0; i < peers_.size(); i++) {
      Peer *peer = &peers_[i];
      peer->mutex.Lock();
      peer->connection.PrepareSelect(&read_fds, &write_fds, &error_fds);
      peer->mutex.Unlock();
      maxfd = max(maxfd, peer->connection.read_fd());
      maxfd = max(maxfd, peer->connection.write_fd());
    }

    FD_ZERO(&error_fds);

    // Use a one-second timeout so we can poll for should_stop_ to allow
    // graceful shutdown.
    struct timeval tv;
    tv.tv_sec = 1;
    tv.tv_usec = 0;
    int n_events = select(maxfd + 1, &read_fds, &write_fds, &error_fds, &tv);
    //fprintf(stderr, "%d: select() returns %d\n", rank_, n_events);

    //for (int i = 0; i < maxfd; i++) {
    //  if (FD_ISSET(i, &read_fds)) { //fprintf(stderr, "%d: read on %d\n", rank_, i); }
    //  if (FD_ISSET(i, &write_fds)) { //fprintf(stderr, "%d: write on %d\n", rank_, i); }
    //  if (FD_ISSET(i, &error_fds)) { //fprintf(stderr, "%d: error on %d\n", rank_, i); }
    //}

    if (n_events < 0) {
      NONFATAL("Select failed");
      continue;
    }
    if (n_events == 0) {
      // Select timed out, let's try again
      continue;
    }

    if (FD_ISSET(alert_slot_fd_, &read_fds)) {
      // We got a wake-up signal.  Clear the buffer so we won't receive the
      // signal twice.
      char buf[8];
      while (read(alert_slot_fd_, buf, sizeof(buf)) > 0) {}
    }

    if (FD_ISSET(listen_fd_, &read_fds)) {
      //fprintf(stderr, "%d: Connection.\n", rank_);
      // Accept incoming connections
      for (;;) {
        sockaddr_in addr;
        socklen_t len = sizeof(addr);
        int new_fd;

        mem::Zero(&addr);
        new_fd = accept(listen_fd_, (struct sockaddr*)&addr, &len);

        if (new_fd < 0) {
          break;
        }

        index_t i;

        for (i = 0; i < peers_.size(); i++) {
          Peer *peer = &peers_[i];
          // We don't have to lock here since sin_addr is not locked
          // TODO: Consider moving the locks elsewhere
          if (addr.sin_addr.s_addr
              == peer->connection.peer_addr().sin_addr.s_addr) {
            peer->mutex.Lock();
            peer->connection.AcceptIncoming(new_fd);
            peer->mutex.Unlock();
            //fprintf(stderr, "%d: accepted rank %d on fd %d\n", rank_, i, new_fd);
            break;
          }
        }
        if (i == peers_.size()) {
          NONFATAL("Incomming connection from unknown machine, %s.",
              inet_ntoa(addr.sin_addr));
          (void)close(new_fd);
        }
      }
    }

    work_items.Resize(0);

    mutex_.Lock(); // we'll be accessing channels during this time
    // Gather available messages
    for (index_t i = 0; i < peers_.size(); i++) {
      Peer *peer = &peers_[i];

      // We should look for a faster way of scanning file descriptors,
      // especially not having to lock all the mutexes.
      // How about a DenseIntMap?  Or, we can have the mutex be part of the
      // connection itself?
      peer->mutex.Lock();
      // we'll allow errors to occur if we're shutting down.
      peer->connection.HandleSocketEvents(&read_fds, &write_fds, &error_fds,
        status_ != RUN);
      GatherReadyMessages_(peer, &work_items);
      peer->mutex.Unlock();
    }
    mutex_.Unlock();

    // Execute work items while we aren't holding any mutexes
    for (index_t i = 0; i < work_items.size(); i++) {
      WorkItem *item = &work_items[i];
      //fprintf(stderr, "Executing %d:%d\n", item->message->channel(), item->message->transaction_id());
      // it is transaction's responsibility to delete the message
      item->transaction->HandleMessage(item->message);
    }
  }
}

void RpcSockImpl::GatherReadyMessages_(Peer *peer,
    ArrayList<WorkItem> *work_items) {
  ArrayList<Message*>* queue = &peer->connection.read_queue();
  index_t j = 0;
  index_t i;

  for (i = 0; i < queue->size(); i++) {
    Message *message = (*queue)[i];
    int id = message->transaction_id();
    Transaction *transaction;

    if (message->channel() < 0) {
      // When the channel ID is invalid, this means that I was the initiator
      // of the transaction.
      transaction = peer->outgoing_transactions[id];
      DEBUG_ASSERT(transaction != NULL);
    } else {
      // When the channel ID is valid, it means the remote host initiated
      // the transaction and was picked up by my channel server.
      transaction = peer->incoming_transactions[id];
      if (!transaction) {
        // No existing transaction.  We use the channel number to create one.
        Channel *channel = channels_[message->channel()];
        if (channel) {
          transaction = channel->GetTransaction(message);
          transaction->TransactionHandleNewSender_(message);
          peer->incoming_transactions[id] = transaction;
          //fprintf(stderr, "channel %d found, %p\n", message->channel(), transaction);
        } else {
          //fprintf(stderr, "channel %d not found, msg %p\n", message->channel(), message);
        }
      }
    }

    if (transaction != NULL) {
      // This work item is processable, add it to the transactions
      WorkItem *item = work_items->AddBack();
      item->message = message;
      item->transaction = transaction;
    } else {
      // No good... we have to enqueue it.
      (*queue)[j++] = message;
    }
  }
  if (j != i) {
    queue->Resize(j);
  }
}

//-------------------------------------------------------------------------

RpcSockImpl::Peer::Peer() {
  incoming_transactions.Init();
  incoming_transactions.default_value() = NULL;
  outgoing_transactions.Init();
  outgoing_transactions.default_value() = NULL;
}

RpcSockImpl::Peer::~Peer() {
}

//-------------------------------------------------------------------------

void Transaction::Init(int channel_in) {
  // Set up our internal data.
  channel_ = channel_in;
  peers_.Init();
}

Message *Transaction::CreateMessage(int peer, size_t size) {
  Message *message;
  int i;
  int transaction_id;

  // See if we are already dealing with this peer, so that we use the existing
  // transaction ID.
  for (i = 0; i < peers_.size(); i++) {
    if (peers_[i].peer == peer) {
      break;
    }
  }

  if (i == peers_.size()) {
    // We haven't sent or received from this peer, so we need to send the
    // channel number to it so that the channel can create a new transaction.
    peers_.AddBack();
    transaction_id = RpcSockImpl::instance.AssignTransaction(peer, this);
    peers_[i].peer = peer;
    peers_[i].channel = channel();
    peers_[i].transaction_id = transaction_id;
  }

  // Create a message we can send!
  message = SockConnection::CreateMessage(
      peers_[i].peer, peers_[i].channel, peers_[i].transaction_id, size);

  return message;
}

void Transaction::TransactionHandleNewSender_(Message *message) {
  // We got a message from a new sender.

  // We'll reply to this with channel -1, meaning that it was the other end
  // who initiated the transaction ID, i.e., the transaction ID lives in
  // their namespace.
  PeerInfo *peer_info = peers_.AddBack();
  peer_info->peer = message->peer();
  peer_info->channel = -1;
  peer_info->transaction_id = message->transaction_id();
}

void Transaction::Send(Message *message) {
  // RpcSockImpl knows how to send messages, we don't need to bother with it.
  RpcSockImpl::instance.Send(message);
}

void Transaction::Done() {
  for (index_t i = 0; i < peers_.size(); i++) {
    RpcSockImpl::instance.UnregisterTransaction(
        peers_[i].peer, peers_[i].channel, peers_[i].transaction_id);
  }
  peers_.Clear();
}

void Transaction::Done(int peer) {
  for (index_t i = 0; i < peers_.size(); i++) {
    // TODO: Demeter?
    if (peer == peers_[i].peer) {
      RpcSockImpl::instance.UnregisterTransaction(
          peers_[i].peer, peers_[i].channel, peers_[i].transaction_id);
      peers_[i] = peers_[peers_.size()-1];
      peers_.PopBack();
      break;
    }
  }
}

//-------------------------------------------------------------------------

SockConnection::~SockConnection() {
  // TODO: There are more socket functions I might have to call
  if (is_read_open()) {
    (void) close(read_fd_);
  }
  if (is_write_open()) {
    (void) close(write_fd_);
  }
}

Message *SockConnection::CreateMessage(
    int peer, int channel, int transaction_id, size_t size) {
  Message *message = new Message();
  char *buffer = mem::Alloc<char>(sizeof(Header) + size);
  Header *header = reinterpret_cast<Header*>(buffer);

  message->Init(peer, channel, transaction_id, buffer,
      sizeof(Header), size);
  header->magic = MAGIC;
  header->channel = message->channel();
  header->transaction_id = message->transaction_id();
  header->data_size = message->data_size();

  return message;
}

void SockConnection::Init(int peer_num, const char *ip_address, int port) {
  peer_ = peer_num;
  
  mem::Zero(&peer_addr_);
  peer_addr_.sin_family = AF_INET;
  peer_addr_.sin_port = htons(port);
  if (inet_pton(AF_INET, ip_address, &peer_addr_.sin_addr) < 0) {
    FATAL("Invalid IP address [%s] -- must be 1.2.3.4 format\n", ip_address);
  }

  read_total_ = 0;
  read_message_ = NULL;
  read_buffer_pos_ = BIG_BAD_NUMBER;
  read_queue_.Init();

  write_total_ = 0;
  write_message_ = NULL;
  write_buffer_pos_ = BIG_BAD_NUMBER;
  write_queue_.Init();

  read_fd_ = -1;
  write_fd_ = -1;
}

void SockConnection::OpenOutgoing() {
  write_fd_ = socket(AF_INET, SOCK_STREAM, 0);

  MakeSocketNonBlocking(write_fd_);

  //fprintf(stderr, "connect to peer %d, %s\n", peer_, inet_ntoa(peer_addr_.sin_addr));
  if (0 > connect(write_fd_, (struct sockaddr*)&peer_addr_, sizeof(struct sockaddr_in))
      && errno != EINTR && errno != EINPROGRESS) {
    FATAL("connect failed: %s", strerror(errno));
  }

  // We'd nominally have to wake up the polling loop here to inform the
  // loop that we've just made a connection -- however, no need, because
  // Send will wake up the polling loop since write_message_ is currently
  // NULL.
}

void SockConnection::AcceptIncoming(int accepted_fd) {
  read_fd_ = accepted_fd;
  MakeSocketNonBlocking(read_fd_);
}

void SockConnection::Send(Message *message) {
  if (!is_write_open()) {
    // Open our outgoing link if one doesn't exist
    OpenOutgoing();
  }

  ++write_total_;
  if (likely(write_message_ == NULL)) {
    // If we're not writing anything now, set our current message.
    write_buffer_pos_ = 0;
    write_message_ = message;
    RpcSockImpl::instance.WakeUpPollingLoop();
  } else {
    // We're already writing something, put it on the priority queue
    write_queue_.Put(write_total_, message);
  }
}

void SockConnection::TryWrite() {
  while (is_writing()) {
    if (write_buffer_pos_ == write_message_->buffer_size()) {
      // Looks like we successfully wrote the whole message.
      //fprintf(stderr, "Wrote packet %d\n", write_message_->transaction_id());
      delete write_message_;
      if (write_queue_.is_empty()) {
        write_message_ = NULL;
        break;
      } else {
        write_buffer_pos_ = 0;
        write_message_ = write_queue_.Pop();
      }
    }
    // Try to write something.
    ssize_t bytes_written = write(write_fd_,
        write_message_->buffer() + write_buffer_pos_,
        write_message_->buffer_size() - write_buffer_pos_);
    //fprintf(stderr, "WRITE %d bytes\n", bytes_written);
    if (bytes_written <= 0) {
      // Okay, we weren't able to write anything.
      if (bytes_written < 0 && errno != EAGAIN && errno != EINTR) {
        // It turns out the error is not just a non-blocking type error,
        // so we die and let the entire team die out too.
        FATAL("Error writing");
      }
      return;
    } else if (bytes_written != 0) {
      // We successfully wrote something, update our position.
      write_buffer_pos_ += bytes_written;
    }
  }
}

void SockConnection::TryRead() {
  //fprintf(stderr, "Trying to read!\n");
  for (;;) {
    // First, read a header if we have to.
    if (!read_message_) {
      // It looks like we weren't in the process of reading another packet, so
      // this is a new packet.
      Header header;
      // Read the packet's header.
      ssize_t header_bytes = read(read_fd_, &header, sizeof(Header));
      // Error out if we got the wrong number of bytes.
      // In theory, we probably can't always assume the headers won't be be
      // chopped up.
      if (header_bytes != sizeof(Header)) {
        if (header_bytes == 0 || errno == EINTR || errno == EAGAIN) {
          // okay, it looks like there isn't any data
          //fprintf(stderr, "Looks like we don't actually have data...\n");
          return;
        } else {
          FATAL("Error reading packet header: read returned %d bytes: %s",
              int(header_bytes), strerror(errno));
        }
      }
      DEBUG_SAME_INT(header.magic, MAGIC);
      // When we read in a message, we don't need to allocate space for the
      // header (since we have already read it successfully).
      read_message_ = new Message();
      read_message_->Init(peer_, header.channel, header.transaction_id,
          mem::Alloc<char>(header.data_size), 0, header.data_size);
      read_buffer_pos_ = 0;
      //fprintf(stderr, "Read header %d bytes\n", header_bytes);
      //fprintf(stderr, "Got a valid header.\n");
    }
    // Second, see if we're done with the packet.  (Note some packets have
    // a null message length!)
    if (read_buffer_pos_ == read_message_->buffer_size()) {
      // We've read a whole message.  Put it on the queue to be serviced.
      ++read_total_;
      *read_queue_.AddBack() = read_message_;
      //fprintf(stderr, "Got packet %d:%d\n", read_message_->channel(), read_message_->transaction_id());

      read_message_ = NULL;
      read_buffer_pos_ = 0;
      break;
    }
    // Finally, read as much payload as we can for this message.
    ssize_t bytes_read = read(read_fd_,
        read_message_->buffer() + read_buffer_pos_,
        read_message_->buffer_size() - read_buffer_pos_);
    //fprintf(stderr, "Got %d data bytes.\n", (int)bytes_read);
    //fprintf(stderr, "Read payload %d bytes\n", bytes_read);
    if (bytes_read > 0) {
      read_buffer_pos_ += bytes_read; 
    } else {
      // Couldn't read anything.
      if (bytes_read != 0 && errno != EAGAIN && errno != EINTR) {
        // Error wasn't due to the fact that it's non-blocking, so the socket
        // was disconnected.
        FATAL("Error reading");
      }
      break;
    }
  }
}

void SockConnection::HandleSocketEvents(
    fd_set *read_fds, fd_set *write_fds, fd_set *error_fds,
    bool allow_errors) {
  /*
   code to read socket errors:
      int sockError = 0;
      socklen_t sockErrorLen = sizeof(sockError);
      if (getsockopt(sock, SOL_SOCKET, SO_ERROR,
        &sockError, &sockErrorLen) == -1)
      ...
  */
  if (unlikely(is_read_open())) {
    if (FD_ISSET(read_fd_, error_fds)) {
      // Poor man's way to terminate all processes
      if (allow_errors) {
        read_fd_ = -1;
      } else {
        FATAL("Socket error on read fd");
      }
    }
    if (FD_ISSET(read_fd_, read_fds)) {
      TryRead();
    }
  }
  if (unlikely(is_write_open())) {
    if (FD_ISSET(write_fd_, error_fds)) {
      // Poor man's way to terminate all processes
      if (allow_errors) {
        write_fd_ = -1;
      } else {
        FATAL("Socket error on out fd");
      }
    }
    if (FD_ISSET(write_fd_, write_fds)) {
      TryWrite();
    }
  }
}

void SockConnection::PrepareSelect(
    fd_set *read_fds, fd_set *write_fds, fd_set *error_fds) {
  if (is_read_open()) {
    FD_SET(read_fd_, read_fds);
    FD_SET(read_fd_, error_fds);
  }
  if (is_write_open()) {
    if (is_writing()) {
      FD_SET(write_fd_, write_fds);
    }
    FD_SET(write_fd_, error_fds);
  }
}

/**
 * @file rpc_sock.cc
 *
 * Implementation of transaction API using TCP.
 */

#include "rpc.h"
#include "rpc_sock.h"

#include "file/textfile.h"

#include <fcntl.h>
#include <errno.h>
#include <string.h>

//-------------------------------------------------------------------------

#warning "avoid transaction messages"
#warning "put buffer limit"
#warning "use udp"

RpcSockImpl::Cleanup RpcSockImpl::cleanup_;

/*
in the future:
 - fault tolerance
 - smarter "initialize" that doesn't require the port to be chosen in
 advance

tasks to complete that must work
 /- smarter transaction lookup so barriers can all share the same number
 /- smarter fd_set usage (perhaps a simple linear scan,
     with fd-lookup table?)
 /- startup and shutdown
 /- integrate with rest of code
   /- rpc calls
   /- rpc servers
 /- barriers - birth and death
 /- name resolution
   /- REQUIRE IP ADDRESS - Duhh! :-)

near future
 /- abstract reduce operation that uses the tree structure

things to think about
 - how to shut down cleanly and respond to signals
   - ssh within the c code or ssh externally?
     - try within c code
   - failures
     - use ssh to detect return code
     - if ssh fails, emit an error message and kill yourself
 - fastexec integration
 /- accept() - send me your rank? or just ignore rank altogether?


resolve address - getaddrinfo
register port numbers
*/

//-------------------------------------------------------------------------

/**
 * The timeout to establish initial connections.
 *
 * This should be on the order of the maximum startup time, inclusing SSH,
 * NFS, and all other goodies -- which can be quite slow at times.
 * If it takes too long to start up, one of the machines is probably dead,
 * and we need to make sure we don't leave ghost processes on random machines.
 */
#define TIMEOUT_INITIAL 90
/**
 * The maximum amount of time to process messages.
 *
 * Message processing should never really do any I/O
 * (except for disk accesses) or do anything that might take a long time,
 * because this blocks other requests from being serviced.
 * This mandates that even if an infinite loop occurs while servicing a
 * request, a ghost process won't be left on the machine.
 */
#define TIMEOUT_MESSAGE 30
/**
 * The maximum amount of time to connect to the parent.
 *
 * This should be on the order of the maximum amount of time it should take
 * a process to start up.
 * This ensures that if the parent dies before we can connect to it, our
 * process will eventually die (and cause others to die cascading with it).
 */
#define TIMEOUT_CONNECT 90
/**
 * The maximum amount of idle time after which pings should be sent to see if
 * our neighbors have died, and this also corresponds to the maximum ping
 * timeout.
 *
 * Ensures that if other processes die, we cascade and also die too -- the
 * tree structure results in all processes dying.
 */
#define TIMEOUT_SELECT 30

namespace {
  void MakeSocketNonBlocking(int fd) {
    fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK);
  }
};

//-------------------------------------------------------------------------

RpcSockImpl *RpcSockImpl::instance = NULL;

void RpcSockImpl::Init() {
  channels_.Init();
  channels_.default_value() = NULL;
  unknown_connections_.Init();
  live_pings_ = 0;
  peer_from_fd_.Init();
  peer_from_fd_.default_value() = -1;
  max_fd_ = 0;
  unacknowledged_ = 0;
  FD_ZERO(&error_fd_set_);
  FD_ZERO(&read_fd_set_);
  FD_ZERO(&write_fd_set_);

  module_ = fx_submodule(fx_root, "rpc", "rpc");

  if (!fx_param_exists(module_, "n")) {
    n_peers_ = 1;
    rank_ = 0;
    port_ = -1;
  } else {
    n_peers_ = fx_param_int_req(module_, "n");
    rank_ = fx_param_int(module_, "rank", 0);
    DEBUG_ASSERT(rank_ < n_peers_);
    port_ = fx_param_int(module_, "port", 31415);
  }

  CreatePeers_();
  CalcChildren_();

  if (n_peers_ != 1) {
    Listen_();
    fprintf(stderr, "rpc_sock(%d): Ready, listening on port %d\n", rpc::rank(), port_);

    status_ = INIT;

    if (!rpc::is_root()) {
      peers_[rpc::parent()].mutex.Lock();
      peers_[rpc::parent()].connection.OpenOutgoing(true);
      peers_[rpc::parent()].mutex.Unlock();
    }

    StartPollingThread_();

    // Have an initial barrier to make sure all processors are alive.
    rpc::Barrier(0);
    status_ = RUN;

    fprintf(stderr, "rpc_sock(%d): All computers are alive -- starting!\n", rpc::rank());
  }
}

void RpcSockImpl::Done() {
  status_ = STOP_SYNC;
  if (n_peers_ != 1) {
    // Synchronize all machines, to make sure that they're all ready to stop.
    rpc::Barrier(1);
    status_ = STOP;
    WakeUpPollingLoop();
    polling_thread_.WaitStop();
    close(listen_fd_);
    peers_.Resize(0); // automatically calls their destructors
  }
}

void RpcSockImpl::WriteFlush() {
  fd_mutex_.Lock();
  while (unacknowledged_ != 0) {
    flush_cond_.Wait(&fd_mutex_);
  }
  fd_mutex_.Unlock();
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
  bool control_message = unlikely(message->transaction_id() == TID_CONTROL);

  DEBUG_ASSERT_MSG(n_peers_ != 1,
     "I'm the only machine -- why am I trying to send messages over the network?");

  if (!control_message) {
    fd_mutex_.Lock();
    // This used to make sure the buffer pool didn't get too big, but this
    // causes problems within the server thread.  For now, the distributed
    // cache will handle it.
    unacknowledged_++;
    fd_mutex_.Unlock();
  }

  Peer *peer = &peers_[message->peer()];
  peer->mutex.Lock();
  peer->connection.RawSend(message);
  peer->mutex.Unlock();
}

void RpcSockImpl::WakeUpPollingLoop() {
  // Write anything to our alert socket to wake up the network thread.
  if (n_peers_ != 1) {
    (void) write(alert_signal_fd_, "x", 1);
  }
}

void RpcSockImpl::UnregisterTransaction(int peer_num, int channel, int id) {
  Peer *peer = &peers_[peer_num];

  peer->mutex.Lock(); // Lock peer's mutex
  if (channel < 0) {
    DEBUG_ASSERT(peer->incoming_transactions.get(id) != NULL);
    peer->incoming_transactions[id] = NULL;
  } else {
    //fprintf(stderr, "%d to %d: unregistering %d\n", rpc::rank(), peer_num, id);
    DEBUG_ASSERT_MSG(peer->outgoing_transactions[id] != NULL,
       "%d: %d of %d already NULL when unregistering outgoing transaction",
       rpc::rank(), id, peer->outgoing_transactions.size());
    peer->outgoing_transactions[id] = NULL;
    peer->outgoing_freelist[id] = peer->outgoing_free;
    peer->outgoing_free = id;
  }
  peer->mutex.Unlock();
}

int RpcSockImpl::AssignTransaction(int peer_num, Transaction *transaction) {
  Peer *peer = &peers_[peer_num];
  int id;
  peer->mutex.Lock();
  id = peer->outgoing_free;
  if (id >= 0) {
    peer->outgoing_free = peer->outgoing_freelist[id];
  } else {
    id = peer->outgoing_transactions.size();
    peer->outgoing_transactions.Resize(id + 1);
    peer->outgoing_freelist.Resize(id + 1);
  }
  //fprintf(stderr, "%d to %d: registering %d\n", rpc::rank(), peer_num, id);
  DEBUG_ONLY(peer->outgoing_freelist[id] = -1);
  peer->outgoing_transactions[id] = transaction;
  peer->mutex.Unlock();
  return id;
}

//-- helpers for transaction stuff ----------------------------------------


//-- helper functions for initialization and the main loop

void RpcSockImpl::CreatePeers_() {

  peers_.Init(n_peers_);

  if (n_peers_ == 1) {
    Peer *peer = &peers_[0];
    peer->connection.Init(0, "127.0.0.1", -1);
  } else {
    TextLineReader reader;
    reader.Open(fx_param_str_req(module_, "peers"));

    for (index_t i = 0; i < peers_.size(); i++) {
      Peer *peer = &peers_[i];

      peer->connection.Init(i, reader.Peek().c_str(), port_);
      reader.Gobble();
    }
  }
}

void RpcSockImpl::CalcChildren_() {
  int m = 1
      + min(unsigned(n_peers_ - rank_ - 1), unsigned((~rank_) & (rank_-1)));
  int i;

  children_.Init();

  // okay, all peers between my rank and my rank + m - 1 are my direct or
  // indirect children.  the ones that have a power of two difference from
  // me are my direct children.

  // Find the largest power of two difference.
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

  // Open a point-to-point unix socket to myself.
  // We'll use this socket for waking up the network thread.
  socketpair(AF_LOCAL, SOCK_STREAM, 0, sv);
  alert_signal_fd_ = sv[0];
  alert_slot_fd_ = sv[1];
  MakeSocketNonBlocking(alert_signal_fd_);
  MakeSocketNonBlocking(alert_slot_fd_);
  RegisterReadFd(-1, alert_slot_fd_);

  // Create a file descriptor we'll use to listen to sockets.
  listen_fd_ = socket(PF_INET, SOCK_STREAM, 0);
  mem::Zero(&my_address);
  my_address.sin_family = AF_INET;
  my_address.sin_port = htons(port_);
  my_address.sin_addr.s_addr = htonl(INADDR_ANY);

  // Ports normally stay "ghosted" for a specified amount of time after a
  // process finishes.  This allows us to reuse the port immediately instead
  // so you don't have to manually rotate ports.
  int sol_value = 1;
  if (0 > setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR,
      &sol_value, sizeof(sol_value))) {
    NONFATAL("Could not set socket to allow reuse.");
  }

  if (0 > bind(listen_fd_, (struct sockaddr*)&my_address, sizeof(my_address))) {
    // This will fail if the port is in use.
    // TODO: A higher-quality implementation would automatically select a port
    // based on availability, and send this to the master machine.
    FATAL("Could not bind to selected port %d on fd %d: %s",
        port_, listen_fd_, strerror(errno));
  }

  // Try to listen.  This shouldn't ever fail.
  if (0 > listen(listen_fd_, 10)) {
    FATAL("listen() failed, port %d", port_);
  }

  MakeSocketNonBlocking(listen_fd_);
  RegisterReadFd(-1, listen_fd_);
}

void RpcSockImpl::StartPollingThread_() {
  polling_task_.Init(this);
  polling_thread_.Init(&polling_task_);
  polling_thread_.Start();
}

void RpcSockImpl::PollingLoop_() {
  ArrayList<WorkItem> work_items;
  fd_set read_fds;
  fd_set write_fds;
  fd_set error_fds;
  int initialization_seconds = 0;

  work_items.Init();

  while (status_ != STOP) {
    index_t j = 0;

    fd_mutex_.Lock();
    read_fds = read_fd_set_;
    write_fds = write_fd_set_;
    error_fds = error_fd_set_;
    fd_mutex_.Unlock();

    struct timeval tv;
    tv.tv_sec = TIMEOUT_SELECT;
    tv.tv_usec = 0;
    int n_events = select(max_fd_ + 1, &read_fds, &write_fds, &error_fds, &tv);

    if (n_events < 0) {
      FATAL("%d: select() failed: %s", rank_, strerror(errno));
    } else if (n_events == 0) {
      // Select timed out, send ping messages to test the connections
      if (status_ == INIT) {
        initialization_seconds += TIMEOUT_SELECT;
        if (initialization_seconds >= TIMEOUT_INITIAL) {
          FATAL("Initialization took longer than %d seconds, dying.",
              TIMEOUT_INITIAL);
        }
      } else {
        // The ping method isn't really sophisticated -- if some of our peers
        // are still alive, we probably won't bother to start pinging anyone.
        // Chances are though that the death of another machine would cause
        // things to crawl to a halt, and we'll eventually start sending
        // pings.
        if (live_pings_ != 0) {
          FATAL("rpc_sock(%d): Ping timeout to %d peers.",
              rpc::rank(), live_pings_);
        }
        if (!rpc::is_root()) {
          Ping_(rpc::parent(), MSG_PING);
        }
        for (index_t i = 0; i < rpc::n_children(); i++) {
          Ping_(rpc::child(i), MSG_PING);
        }
      }
    } else {
      if (FD_ISSET(alert_slot_fd_, &read_fds)) {
        // We got a wake-up signal.  Clear the buffer to avoid re-trigger.
        char buf[32];
        while (read(alert_slot_fd_, buf, sizeof(buf)) > 0) {}
      }

      if (unlikely(unknown_connections_.size() != 0)) {
        // Survey any connections that we accepted on faith.
        j = 0;
        for (index_t i = 0; i < unknown_connections_.size(); i++) {
          int fd = unknown_connections_[i];

          if (FD_ISSET(fd, &read_fds)) {
            TryAcceptConnection_(fd);
            FD_CLR(fd, &read_fds);
          } else {
            unknown_connections_[j++] = unknown_connections_[i];
          }
        }
        unknown_connections_.Resize(j);
      }

      bool errors_ok = (status_ == STOP) || (status_ == STOP_SYNC);

      for (int i = max_fd_; i >= 0; i--) {
        if (unlikely(FD_ISSET(i, &error_fds))) {
          if (!errors_ok) {
            FATAL("Unexpected error on file descriptor %d (i'm %d)\n",
                i, rpc::rank());
          }
          FD_CLR(i, &error_fd_set_);
          FD_CLR(i, &read_fd_set_);
          DeactivateWriteFd(i);
        }

        if (unlikely(FD_ISSET(i, &read_fds))) {
          int peer_num = peer_from_fd_[i];
          if (peer_num >= 0) {
            Peer *peer = &peers_[peer_num];
            peer->mutex.Lock();
            if (!peer->connection.TryRead() && !errors_ok) {
              FATAL("Unexpected end of file for peer %d (i'm %d)",
                  peer_num, rpc::rank());
            }
            peer->is_pending = true;
            peer->mutex.Unlock();
          }
        }

        if (FD_ISSET(i, &write_fds)) {
          int peer_num = peer_from_fd_[i];
          if (peer_num >= 0) {
            Peer *peer = &peers_[peer_num];
            peer->mutex.Lock();
            peer->connection.TryWrite();
            peer->mutex.Unlock();
          }
        }
      }

      mutex_.Lock();
      alarm(TIMEOUT_MESSAGE);
      for (index_t i = 0; i < peers_.size(); i++) {
        // no mutexes required here - the 'pending' field is only modified
        // by the network thread
        Peer *peer = &peers_[i];
        if (unlikely(peer->is_pending)) {
          ExecuteReadyMessages_(peer);
        }
      }
      alarm(0);
      mutex_.Unlock();

      fd_mutex_.Lock();
      if (unacknowledged_ == 0) {
        flush_cond_.Broadcast();
      }
      fd_mutex_.Unlock();

      if (unlikely(FD_ISSET(listen_fd_, &read_fds))) {
        // Accept incoming connections
        int new_fd;
        while ((new_fd = accept(listen_fd_, NULL, NULL)) >= 0) {
          *unknown_connections_.AddBack() = new_fd;
          RegisterReadFd(-1, new_fd);
        }
      }
    }
  }
}

void RpcSockImpl::TryAcceptConnection_(int fd) {
  SockConnection::Header preheader;

  // TODO: This protocol message is unusual in that the transaction ID is
  // being misused as the peer number.  Ideally, the peer number should
  // be stored as data() in the message.
  ssize_t read_result = read(fd, &preheader, sizeof(preheader));
  
  if (read_result != sizeof(preheader)) {
    NONFATAL("%d: Rejected connection -- incomplete header - %d.", rank_, int(read_result));
  } else if (preheader.magic != SockConnection::MAGIC) {
    NONFATAL("%d: Rejected connection -- bad magic number.", rank_);
  } else if (preheader.channel != SockConnection::BIRTH_CHANNEL) {
    NONFATAL("%d: Rejected connection -- channel is not the birth channel.", rank_);
  } else if (preheader.data_size != 0) {
    NONFATAL("%d: Rejected connection -- bad payload size for birth message.", rank_);
  } else if (preheader.transaction_id < 0
      || preheader.transaction_id >= n_peers_) {
    NONFATAL("%d: Rejected connection -- peer number (trans ID) is out of range.", rank_);
  } else {
    int peer_num = preheader.transaction_id; // use trans ID for peer #
    Peer *peer = &peers_[peer_num];

    peer->mutex.Lock();
    peer->connection.AcceptIncoming(fd);
    peer->mutex.Unlock();

    RegisterReadFd(peer_num, fd);

    return;
  }

  (void) close(fd);
}

void RpcSockImpl::ExecuteReadyMessages_(Peer *peer) {
  int handled = 0;

  // no mutex required on read_queue, pending, or is_pending, because the
  // network thread is the only one that deals with these

  while (!peer->connection.read_queue().is_empty()) {
    Message *message = peer->connection.read_queue().Pop();
    int id = message->transaction_id();
    Transaction *transaction;

    if (id < 0) {
      DEBUG_ASSERT_MSG(id == TID_CONTROL,
          "Negative transaction ID's should correspond to control messages.");
      if (message->channel() == MSG_PONG) {
        live_pings_--;
      } else if (message->channel() == MSG_PING) {
        Ping_(peer->connection.peer(), MSG_PONG);
      } else if (message->channel() == MSG_ACK) {
        fd_mutex_.Lock();
        unacknowledged_ -= *message->data_as<int32>();
        DEBUG_ASSERT_MSG(unacknowledged_ >= 0,
            "Received more acknowledgements than messages sent.");
        fd_mutex_.Unlock();
      } else {
        FATAL("Unknown control message: %d", message->channel());
      }
    } else if (message->channel() < 0) {
      // When the channel ID is invalid, this means that I was the initiator
      // of the transaction.  Let's handle this immediately.
      peer->mutex.Lock();
      transaction = peer->outgoing_transactions[id];
      DEBUG_ASSERT_MSG(transaction != NULL,
         "Transaction null, channel %d, id %d, me %d",
         message->channel(), id, rpc::rank());
      peer->mutex.Unlock();
      transaction->HandleMessage(message);
      handled++;
    } else {
      peer->pending.Add(message);
    }
  }

  peer->is_pending = false;

  while (!peer->pending.is_empty()) {
    Message *message = peer->pending.top();
    int id = message->transaction_id();
    // When the channel ID is valid, it means the remote host initiated
    // the transaction.
    peer->mutex.Lock();
    Transaction *transaction = peer->incoming_transactions[id];
    if (!transaction) {
      // No existing transaction.  We use the channel number to create one.
      Channel *channel = channels_[message->channel()];
      if (channel) {
        transaction = channel->GetTransaction(message);
        transaction->TransactionHandleNewSender_(message);
        peer->incoming_transactions[id] = transaction;
      }
    }
    peer->mutex.Unlock();

    if (!transaction) {
      peer->is_pending = true;
      break;
    }

    handled++;
    transaction->HandleMessage(message);
    peer->pending.Pop();
  }

  if (handled != 0) {
    // send acknowledgement of the messages we've processed
    Message *ack = peer->connection.CreateMessage(MSG_ACK, TID_CONTROL,
        sizeof(int32));
    *ack->data_as<int32>() = handled;
    Send(ack);
  }
}

void RpcSockImpl::Ping_(int peer_num, int number) {
  Peer *peer = &peers_[peer_num];
  peer->mutex.Lock();
  // Only ping machines we're open both ways with.  We don't want to try
  // pinging a machine that we're in the process of connecting to, but might
  // be waiting for a connection to its parent.
  if (peer->connection.is_write_open() && peer->connection.is_read_open()) {
    peer->connection.RawSend(
        peer->connection.CreateMessage(number, TID_CONTROL, 0));
    if (number == MSG_PING) {
      live_pings_++;
    }
  }
  peer->mutex.Unlock();
}

void RpcSockImpl::RegisterReadFd(int peer, int fd) {
  fd_mutex_.Lock();
  FD_SET(fd, &read_fd_set_);
  FD_SET(fd, &error_fd_set_);
  max_fd_.MaxWith(fd);
  if (peer >= 0) {
    peer_from_fd_[fd] = peer;
  }
  fd_mutex_.Unlock();
}

void RpcSockImpl::RegisterWriteFd(int peer, int fd) {
  ActivateWriteFd(fd);
  fd_mutex_.Lock();
  FD_SET(fd, &error_fd_set_);
  max_fd_.MaxWith(fd);
  if (peer >= 0) {
    peer_from_fd_[fd] = peer;
  }
  fd_mutex_.Unlock();
}

void RpcSockImpl::ActivateWriteFd(int fd) {
  fd_mutex_.Lock();
  FD_SET(fd, &write_fd_set_);
  fd_mutex_.Unlock();
}

void RpcSockImpl::DeactivateWriteFd(int fd) {
  fd_mutex_.Lock();
  FD_CLR(fd, &write_fd_set_);
  fd_mutex_.Unlock();
}

//-------------------------------------------------------------------------

RpcSockImpl::Peer::Peer() {
  incoming_transactions.Init();
  incoming_transactions.default_value() = NULL;
  outgoing_transactions.Init();
  outgoing_freelist.Init();
  outgoing_free = -1;
  is_pending = false;
  pending.Init();
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
    transaction_id = RpcSockImpl::instance->AssignTransaction(peer, this);
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
  rpc::Send(message);
}

void Transaction::Done() {
  for (index_t i = 0; i < peers_.size(); i++) {
    RpcSockImpl::instance->UnregisterTransaction(
        peers_[i].peer, peers_[i].channel, peers_[i].transaction_id);
  }
  peers_.Clear();
}

void Transaction::Done(int peer) {
  for (index_t i = 0; i < peers_.size(); i++) {
    // TODO: Demeter?
    if (peer == peers_[i].peer) {
      RpcSockImpl::instance->UnregisterTransaction(
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
  read_buffer_pos_ = 0;
  read_queue_.Init();

  write_total_ = 0;
  write_message_ = NULL;
  write_buffer_pos_ = 0;
  write_queue_.Init();

  read_fd_ = -1;
  write_fd_ = -1;
}

void SockConnection::OpenOutgoing(bool blocking) {
  int temp_fd = socket(PF_INET, SOCK_STREAM, 0);

  if (temp_fd < 0) {
    FATAL("Too many sockets");
  }

  if (blocking) {
    int sleeptime = 1;
    int elapsed_time = 0;

    while (0 > connect(temp_fd, (struct sockaddr*)&peer_addr_,
        sizeof(struct sockaddr_in))) {
      (void) close(temp_fd);

      if (elapsed_time % 10 == 0 && elapsed_time >= 10) {
        NONFATAL(
            "rpc_sock(%d): Connection to parent %d failed, we'll try for %d more seconds.\n",
            rpc::rank(), peer_, TIMEOUT_CONNECT - elapsed_time);
      }

      elapsed_time += sleeptime;

      if (elapsed_time >= TIMEOUT_CONNECT) {
        FATAL("Tried connecting to rank %d for %d seconds, bailing out.",
            TIMEOUT_CONNECT, peer_);
      }

      sleep(sleeptime);

      temp_fd = socket(PF_INET, SOCK_STREAM, 0);
    }

    MakeSocketNonBlocking(temp_fd);
  } else {
    MakeSocketNonBlocking(temp_fd);

    //fprintf(stderr, "connect to peer %d, %s\n", peer_, inet_ntoa(peer_addr_.sin_addr));
    if (0 > connect(temp_fd, (struct sockaddr*)&peer_addr_, sizeof(struct sockaddr_in))
        && errno != EINTR && errno != EINPROGRESS) {
      FATAL("connect failed: %s", strerror(errno));
    }
  }

  write_fd_ = temp_fd;

  // Send the initial birth message
  write_buffer_pos_ = 0;
  write_message_ = CreateMessage(peer_, BIRTH_CHANNEL, rpc::rank(), 0);

  RpcSockImpl::instance->RegisterWriteFd(peer_, write_fd_);
}

void SockConnection::AcceptIncoming(int accepted_fd) {
  if (read_fd_ >= 0) {
    FATAL("Two incoming connections from rank %d!\n", peer_);
  }
  read_fd_ = accepted_fd;
  MakeSocketNonBlocking(read_fd_);
}

void SockConnection::RawSend(Message *message) {
  if (unlikely(!is_write_open())) {
    // Open our outgoing link if one doesn't exist.
    OpenOutgoing(false);
    // Alert network thread that there's a new file descriptor
    RpcSockImpl::instance->WakeUpPollingLoop();
  }

  ++write_total_;
  //fprintf(stderr, "%d: sending message to %d\n", rpc::rank(), peer_);
  if (!is_writing()) {
    // If we're not writing anything currently, set our current message, and
    // try to write it immediately without blocking.
    write_buffer_pos_ = 0;
    write_message_ = message;
    TryWrite();
    if (is_writing()) {
      // We couldn't write the whole message without blocking,
      // so wake up polling loop so we can select on the socket
      // to become writable.
      RpcSockImpl::instance->ActivateWriteFd(write_fd_);
      RpcSockImpl::instance->WakeUpPollingLoop();
    }
  } else {
    // We're already writing something, put it on the priority queue.
    // No need to wake up the polling loop, because it's already quite aware
    // that we want to check if this socket is readable.
    write_queue_.Add(message);
  }
}

void SockConnection::TryWrite() {
  if (is_writing()) {
    for (;;) {
      if (write_buffer_pos_ == write_message_->buffer_size()) {
        // Looks like we successfully wrote the whole message.
        delete write_message_;
        if (write_queue_.is_empty()) {
          write_message_ = NULL;
          RpcSockImpl::instance->DeactivateWriteFd(write_fd_);
          return;
        } else {
          write_message_ = write_queue_.Pop();
          write_buffer_pos_ = 0;
          DEBUG_ASSERT(write_message_->buffer_size() != 0);
        }
      }
      // Try to write something.
      ssize_t bytes_written = write(write_fd_,
          write_message_->buffer() + write_buffer_pos_,
          write_message_->buffer_size() - write_buffer_pos_);
      if (bytes_written <= 0) {
        // Okay, we weren't able to write anything.
        if (bytes_written < 0 && errno != EAGAIN && errno != EINTR) {
          // It turns out the error is not just a non-blocking type error,
          // so we die and let the entire team die out too.
          FATAL("%d: Error writing to %d: %s", rpc::rank(), peer_, strerror(errno));
        }
        return;
      } else if (bytes_written != 0) {
        // We successfully wrote something, update our position.
        write_buffer_pos_ += bytes_written;
      }
    }
  } else {
    RpcSockImpl::instance->DeactivateWriteFd(write_fd_);
  }
}

bool SockConnection::TryRead() {
  //fprintf(stderr, "Trying to read!\n");
  bool anything_done = false;

  for (;;) {
    // First, read a header if we have to.
    if (!read_message_) {
      ssize_t bytes = read(read_fd_,
          reinterpret_cast<char*>(&read_header_) + read_buffer_pos_,
          sizeof(Header) - read_buffer_pos_);

      if (bytes <= 0) {
        if (bytes == 0 || (bytes < 0 && (errno == EINTR || errno == EAGAIN))) {
          // not really an error, just no data
          break;
        } else {
          FATAL("Error reading packet header: read returned %d bytes: %s",
              int(bytes), strerror(errno));
        }
      }

      anything_done = true;
      read_buffer_pos_ += bytes;
      
      if (read_buffer_pos_ == sizeof(Header)) {
        DEBUG_SAME_INT(read_header_.magic, MAGIC);
        // When we read in a message, we don't need to allocate space for the
        // read_header_ (since we have already read it successfully).
        read_message_ = new Message();
        read_message_->Init(peer_, read_header_.channel, read_header_.transaction_id,
            mem::Alloc<char>(read_header_.data_size), 0, read_header_.data_size);
        read_buffer_pos_ = 0;
      } else {
        break;
      }
    }
    // Second, see if we're done with the packet.  (Note some packets have
    // a null message length!)
    if (read_buffer_pos_ == read_message_->buffer_size()) {
      // We've read a whole message.  Put it on the queue to be serviced.
      ++read_total_;
      read_queue_.Add(read_message_);

      read_message_ = NULL;
      read_buffer_pos_ = 0;
      //fprintf(stderr, "%d: got message from %d\n", rpc::rank(), peer_);
      break;
    }
    // Finally, read as much payload as we can for this message.
    ssize_t bytes_read = read(read_fd_,
        read_message_->buffer() + read_buffer_pos_,
        read_message_->buffer_size() - read_buffer_pos_);

    if (bytes_read > 0) {
      anything_done = true;
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
  
  return anything_done;
}


  /*
   code to get a socket error:
      int sockError = 0;
      socklen_t sockErrorLen = sizeof(sockError);
      if (getsockopt(sock, SOL_SOCKET, SO_ERROR,
        &sockError, &sockErrorLen) == -1)
      ...
  */

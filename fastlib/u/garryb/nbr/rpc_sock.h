/**
 * @file rpc_sock.h
 *
 * Remote transaction API using sockets.
 */

#ifndef RPC_SOCK_H
#define RPC_SOCK_H

#include "fastlib/fastlib_int.h"

#include <sys/socket.h>
#include <sys/types.h>
#include <sys/select.h>
#include <sys/time.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>

class Message {
 private:
  int peer_;
  int channel_;
  int transaction_id_;
  char *buffer_;
  char *data_;
  size_t data_size_;

 public:
  Message() {}
  ~Message() {
    mem::Free(buffer_);
  }

  void Init(int peer_in, int channel_in, int transaction_id_in,
      char *buffer_in, size_t header_size_in, size_t data_size_in) {
    peer_ = peer_in;
    channel_ = channel_in;
    transaction_id_ = transaction_id_in;
    buffer_ = buffer_in;
    data_ = buffer_ + header_size_in;
    data_size_ = data_size_in;
  }

  size_t data_size() const {
    return data_size_;
  }
  size_t header_size() const {
    return data_ - buffer_;
  }
  size_t buffer_size() const {
    return (data_ + data_size_) - buffer_;
  }
  char *data() const {
    return data_;
  }
  char *buffer() const {
    return buffer_;
  }
  int peer() const {
    return peer_;
  }
  int channel() const {
    return channel_;
  }
  int transaction_id() const {
    return transaction_id_;
  }
};

class Transaction {
  FORBID_COPY(Transaction);

 private:
  int channel_;
  struct PeerInfo {
    int peer;
    int channel;
    int transaction_id;
  };
  ArrayList<PeerInfo> peers_;

 protected:
  /** Create a message of a specified size, which you will later Send(). */
  Message *CreateMessage(int peer, size_t size);
  /** Send a message */
  void Send(Message *message);
  /** Unregister the transaction from all peers */
  void Done();
  /** Unregister the transaction from just this peer */
  void Done(int peer);

 public:
  Transaction() {}
  virtual ~Transaction() {}
  
  void Init(int channel_num);

  int channel() const {
    return channel_;
  }

  /**
   * The transaction needs to know about the message before HandleMessage
   * is called -- do not call this directly.
   */
  void TransactionHandleNewSender_(Message *message);

  /**
   * Handles an incoming message and changes internal state.
   *
   * @param message incoming message - DELETE THIS WHEN YOU'RE DONE, IT
   *        WILL NOT BE DELETED FOR YOU
   */
  virtual void HandleMessage(Message *message) = 0;
};

/**
 * A Channel is the mechanism for listening for new messages and starting
 * new transactions.
 *
 * A channel must have a number that is unique and the same for all
 * machines.  Do not use channels less than 100 (these are reserved for the
 * implementation).  Re-using channels right after another is usually a bad
 * idea, so if you have two barriers in a row, use different channels. 
 * Fixing this bug correctly is a possible future task :-)
 */
class Channel {
 public:
  Channel() {}
  virtual ~Channel() {}

  virtual Transaction *GetTransaction(Message *message) {
    FATAL("GetTransaction not implemented?");
  }
};

/**
 * Two-way socket connection.
 *
 * Due to possible race conditions, this currently requires a separate
 * incoming and outgoing socket.
 *
 * This class is not thread-safe because it is assumed it will be called
 * from another class in a thread-safe manner.
 */
class SockConnection {
  FORBID_COPY(SockConnection);

 public:
  enum { MAGIC = 314159265 };
  enum { BIRTH_CHANNEL = -60001 };

  struct Header {
    int32 magic;
    int32 channel;
    int32 transaction_id;
    int32 data_size;
  };

 public:
  static Message* CreateMessage(
      int peer, int channel, int transaction_id, size_t size);

 private:
  int peer_;
  int read_fd_;
  int write_fd_;
  struct sockaddr_in peer_addr_;

  int read_total_;
  Message *read_message_;
  size_t read_buffer_pos_;
  ArrayList<Message*> read_queue_;

  int write_total_;
  Message *write_message_;
  size_t write_buffer_pos_;
  MinHeap<int, Message*> write_queue_;

 public:
  SockConnection() {}
  ~SockConnection();

  /** Creates an unopened SocketConnection placeholder. */
  void Init(int peer, const char *ip_address, int port);
  /** Create an outgoing connection for sending messages. */
  void OpenOutgoing(bool blocking);
  /** Accept an incoming connection for receiving messages. */
  void AcceptIncoming(int fd);

  /** Sends a message. */
  void Send(Message *message);

  // some accessors following

  bool is_read_open() const {
    return read_fd_ >= 0;
  }
  bool is_write_open() const {
    return write_fd_ >= 0;
  }
  bool is_reading() const {
    return read_message_ != NULL;
  }
  bool is_writing() const {
    return write_message_ != NULL;
  }
  int read_fd() const {
    return read_fd_;
  }
  int write_fd() const {
    return write_fd_;
  }
  ArrayList<Message*>& read_queue() {
    return read_queue_;
  }
  const ArrayList<Message*>& read_queue() const {
    return read_queue_;
  }
  const struct sockaddr_in& peer_addr() const {
    return peer_addr_;
  }

  // the next functions have to do with non-blocking I/O
  /**
   * A lock-less FD-set setter.
   */
  void FastPrepareSelect(
      fd_set *read_fds, fd_set *write_fds, fd_set *error_fds) {
    // No locking needed: read_fd_ will never close on us unexpectedly
    if (is_read_open()) {
      FD_SET(read_fd_, read_fds);
      FD_SET(read_fd_, error_fds);
    }
    // No locking needed: write_fd_ will never close on us unexpectedly
    if (is_write_open()) {
      if (is_writing()) {
        FD_SET(write_fd_, write_fds);
      }
      FD_SET(write_fd_, error_fds);
    }    
  }
  /** A lock-less FD-set checker. */
  bool FastCheckEvents(
      fd_set *read_fds, fd_set *write_fds, fd_set *error_fds) {
    if (unlikely(is_read_open())) {
      if (unlikely(FD_ISSET(read_fd_, read_fds))
          || unlikely(FD_ISSET(read_fd_, error_fds))) {
        return true;
      }
    }
    if (unlikely(is_write_open())) {
      if (unlikely(FD_ISSET(write_fd_, write_fds))
          || unlikely(FD_ISSET(write_fd_, error_fds))) {
        return true;
      }
    }
    return false;
  }
  /** Handle socket events, given that FastCheckEvents yielded events. */
  void HandleSocketEvents(fd_set *read_fds, fd_set *write_fds,
      fd_set *error_fds, bool allow_errors);
  /** Try reading something from the queue. */
  void TryRead();
  /** Try writing something from the queue. */
  void TryWrite();
};

class RpcSockImpl {
 private:
  /* Standard packet header */
  enum { MSG_BIRTH=-1, MSG_DONE=-2 };
  enum { CHANNEL_BARRIER=-1 };

  /** This must be a multiple of 16 bytes or else alignment might break! */
  struct Header {
    uint32 magic;
    int32 channel;
    uint32 payload_size;
    int32 extra;
  };

  /**
   * My interaction with a a remote host (peer).
   */
  struct Peer {
   public:
    /** Mutual exclusion lock on this peer's data structures. */
    Mutex mutex;
    /** Socket connection */
    SockConnection connection;
    /**
     * Transactions initiated by the peer, where only incoming messages have
     * a non-negative channel number and outgoing ones have a negative
     * channel number.
     */
    DenseIntMap<Transaction*> incoming_transactions;
    /**
     * Transactions initiated by me, where only outgoing messages have
     * a non-negative channel number and incoming messages have a negative
     * channel number.
     */
    DenseIntMap<Transaction*> outgoing_transactions;

   public:
    Peer();
    ~Peer();
  };

  /**
   * Work item structure -- so we can queue up work items while holding
   * the mutexes, and then release all mutexes while we do the actual work.
   */
  struct WorkItem {
    Peer *peer;
    Message *message;
  };

  /**
   * Task structure just so we can run the polling loop in another thread.
   */
  class PollingTask : public Task {
   private:
    RpcSockImpl *main_object_;
   public:
    void Init(RpcSockImpl *main_object_in) {
      main_object_ = main_object_in;
    }
    void Run() {
      main_object_->PollingLoop_();
    }
  };

 public:
  static const int request_header_size = sizeof(Header);
  static const int response_header_size = 0;

 public:
  static RpcSockImpl *instance;

 private:
  struct datanode *module_;
  int rank_;
  int n_peers_;
  uint16 port_;

  int parent_;
  ArrayList<int> children_;

  int listen_fd_;
  ArrayList<Peer> peers_;

  ArrayList<int> unknown_connections_;

  DenseIntMap<Channel*> channels_;

  /**
   * Status: running, synchronizing all other machines to get ready for a
   * stop, and actually stopping.
   */
  enum { RUN, STOP_SYNC, STOP } status_;
  PollingTask polling_task_;
  Thread polling_thread_;

  int barrier_id_;
  int barrier_registrants_;

  RecursiveMutex mutex_;
  
  /** File descriptor for generating alerts */
  int alert_signal_fd_;
  /** File descriptor listening to alerts */
  int alert_slot_fd_;

 public:
  void Init();
  void Done();
  void Register(int channel_num, Channel *channel);
  void Unregister(int channel_num);
  void Send(Message *message);
  void WakeUpPollingLoop();
  void UnregisterTransaction(int peer, int channel, int transaction_id);
  int AssignTransaction(int peer_num, Transaction *transaction);

  int rank() const { return rank_; }
  int n_peers() const { return n_peers_; }
  const ArrayList<int>& children() const { return children_; }
  bool is_root() const { return parent_ == rank_; }
  int parent() const { return parent_; }

 private:
  void CreatePeers_();
  void CalcChildren_();
  void Listen_();
  void StartPollingThread_();
  void PollingLoop_();
  void GatherReadyMessages_(Peer *peer, ArrayList<WorkItem>* work_items);
  void TryAcceptConnection_(int fd);
};

namespace rpc {
  // The RPC-Transaction API is what follows

  /** Initialize everything. */
  inline void Init() {
    DEBUG_ASSERT(RpcSockImpl::instance == NULL);
    RpcSockImpl::instance = new RpcSockImpl();
    RpcSockImpl::instance->Init();
  }
  /** Close all connections, etc */
  inline void Done() {
    DEBUG_ASSERT(RpcSockImpl::instance != NULL);
    RpcSockImpl::instance->Done();
  }
  /** Gets the rank of the current machine */
  inline int rank() {
    return RpcSockImpl::instance->rank();
  }
  /** Gets the total number of peers. */
  inline int n_peers() {
    return RpcSockImpl::instance->n_peers();
  }
  /** Get the i'th child. */
  inline const int child(int i) {
    return RpcSockImpl::instance->children()[i];
  }
  /** Number of broadcast-tree children. */
  inline const index_t n_children() {
    return RpcSockImpl::instance->children().size();
  }
  /** Whether the root of the tree. */
  inline bool is_root() {
    return RpcSockImpl::instance->is_root();
  }
  /** The parent in the broadcast tree topology. */
  inline int parent() {
    return RpcSockImpl::instance->parent();
  }
  /** Register a channel for new transactions. */
  inline void Register(int channel_num, Channel *channel) {
    RpcSockImpl::instance->Register(channel_num, channel);
  }
  /** Unregister a channel for new transactions. */
  inline void Unregister(int channel_num) {
    RpcSockImpl::instance->Unregister(channel_num);
  }
  /** Deliver a message */
  inline void Send(Message *message) {
    RpcSockImpl::instance->Send(message);
  }
};

#endif

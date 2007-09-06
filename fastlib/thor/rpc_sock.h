/**
 * @file rpc_sock.h
 *
 * Remote transaction API using sockets.
 */

#ifndef THOR_RPC_SOCK_H
#define THOR_RPC_SOCK_H

#include "rpc_base.h"

#include "col/arraylist.h"
#include "col/queue.h"
#include "math/math.h"

#include <sys/socket.h>
#include <sys/types.h>
#include <sys/select.h>
#include <sys/time.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>

/**
 * A two-way socket connection.
 *
 * Due race conditions, we use one socket for incoming messages and one
 * for outgoing.  If one had the time and desire, they could resolve the
 * race condition where both sides simultaneously attempt to connect and
 * reject one of the connections.
 *
 * This class is, by design, not thread-safe because it is assumed it will
 * be protected by a mutex.
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
  Message* CreateMessage(int channel, int transaction_id, size_t size) {
    return CreateMessage(peer_, channel, transaction_id, size);
  }

 private:
  int peer_;
  int read_fd_;
  int write_fd_;
  struct sockaddr_in peer_addr_;

  Header read_header_;
  int64 read_total_;
  Message *read_message_;
  size_t read_buffer_pos_;
  // TODO: Implement a dequeue instead of using a heap
  Queue<Message*> read_queue_;

  int64 write_total_;
  Message *write_message_;
  size_t write_buffer_pos_;
  // TODO: Implement a dequeue instead of using a heap
  Queue<Message*> write_queue_;

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
  void RawSend(Message *message);

  // some accessors following
  /** Do we have an incoming socket open? */
  bool is_read_open() const {
    return read_fd_ >= 0;
  }
  /** Do we have an outgoing socket open? */
  bool is_write_open() const {
    return write_fd_ >= 0;
  }
  /** Are we in the process of reading something? */
  bool is_reading() const {
    return read_message_ != NULL;
  }
  /** Are we in the process of writing something? */
  bool is_writing() const {
    return write_message_ != NULL;
  }
  /** The incoming socket file descriptor. */
  int read_fd() const {
    return read_fd_;
  }
  /** The outgoing socket file descriptor. */
  int write_fd() const {
    return write_fd_;
  }
  /** The total number of messages read. */
  int64 read_total() const {
    return read_total_;
  }
  /** The total number of messages written. */
  int64 write_total() const {
    return write_total_;
  }
  /** Messages which have been read but not been processed. */
  Queue<Message*>& read_queue() {
    return read_queue_;
  }
  /** Messages which have been read but not been processed. */
  const Queue<Message*>& read_queue() const {
    return read_queue_;
  }
  /** The socket address of the peer. */
  const struct sockaddr_in& peer_addr() const {
    return peer_addr_;
  }

  /** Try reading something from the queue. */
  bool TryRead();
  /** Try writing something from the queue. */
  void TryWrite();

  /** The rank of the peer. */
  int peer() const {
    return peer_;
  }
};

class RpcSockImpl {
 private:
  enum {
    /** Transaction ID associated with control messages. */
    TID_CONTROL = -1
  };

  enum {
    /** Channel associated with ping control messages. */
    MSG_PING=-102,
    /** Channel associated with pong control messages. */
    MSG_PONG=-103,
    /** Channel associated with acknowledgement control messages. */
    MSG_ACK=-104
  };

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
    ArrayList<Transaction*> outgoing_transactions;
    /**
     * Free list of outgoing transactions ready for reuse.
     */
    ArrayList<int> outgoing_freelist;
    /**
     * An available outgoing transaction, or -1 if none.
     */
    int outgoing_free;
    /**
     * Whether there are any messages that are pending locally because
     * the associated channel does not yet exist.
     */
    bool is_pending;
    /**
     * Any pending incoming messages.  The pending queue is stalled if
     * the channel they're intended for hasn't been found.
     */
    Queue<Message*> pending;

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

  struct Cleanup {
    ~Cleanup() {
      delete RpcSockImpl::instance;
    }
  };

 public:
  static const int request_header_size = sizeof(Header);
  static const int response_header_size = 0;

 public:
  static RpcSockImpl *instance;

 private:
  static Cleanup cleanup_;

 private:
  struct datanode *module_;
  int rank_;
  int n_peers_;
  int live_pings_;
  int port_;

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
  enum { INIT, RUN, STOP_SYNC, STOP } status_;
  PollingTask polling_task_;
  Thread polling_thread_;

  int barrier_id_;
  int barrier_registrants_;

  RecursiveMutex mutex_;

  /** File descriptor for generating alerts */
  int alert_signal_fd_;
  /** File descriptor listening to alerts */
  int alert_slot_fd_;

  /** Mutex on the fd-sets and peer maps. */
  RecursiveMutex fd_mutex_;
  /** Our read file-descriptor set. */
  fd_set read_fd_set_;
  /** Our write file-descriptor set. */
  fd_set write_fd_set_;
  /** Our error file-descriptor set. */
  fd_set error_fd_set_;

  /** Maps file descriptors to peer. */
  DenseIntMap<int> peer_from_fd_;
  /** Largest file descriptor. */
  MinMaxVal<int> max_fd_;

  /** Number of sent messages that haven't ben acknowledged. */
  int unacknowledged_;
  /** Mutex that is locked when writes are pending. */
  WaitCondition flush_cond_;

 public:
  void Init();
  void Done();
  void Register(int channel_num, Channel *channel);
  void Unregister(int channel_num);
  /**
   * Sends a message.
   *
   * If forbid_blocking is false, then this will block if the buffer pool
   * has grown too large.
   */
  void Send(Message *message);
  void WakeUpPollingLoop();
  void UnregisterTransaction(int peer, int channel, int transaction_id);
  int AssignTransaction(int peer_num, Transaction *transaction);

  void RegisterReadFd(int peer, int fd);
  void RegisterWriteFd(int peer, int fd);
  void ActivateWriteFd(int fd);
  void DeactivateWriteFd(int fd);
  void WriteFlush();

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
  void ExecuteReadyMessages_(Peer *peer);
  void TryAcceptConnection_(int fd);
  void Ping_(int peer_num, int message);
};

namespace rpc {
  // The RPC-Transaction API is what follows

  /** Initialize everything. */
  inline void Init() {
    DEBUG_ASSERT(RpcSockImpl::instance == NULL);
    RpcSockImpl::instance = new RpcSockImpl();
    RpcSockImpl::instance->Init();
  }
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
  inline int child(int i) {
    return RpcSockImpl::instance->children()[i];
  }
  /** Number of broadcast-tree children. */
  inline index_t n_children() {
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
  /** Waits for all pending writes to be sent -- important for barriers! */
  inline void WriteFlush() {
    RpcSockImpl::instance->WriteFlush();
  }
};

#endif

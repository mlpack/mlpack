/**
 * @file rpc_base.h
 *
 * Fundamental RPC constructs.
 */

#ifndef THOR_RPC_BASE_H
#define THOR_RPC_BASE_H

#include "../base/base.h"
#include <vector>

/**
 * A Message is the fundamental unit of message-passing in our RPC system.
 *
 * A Message has an optional header (which is used by the underlying
 * RPC delivery system).
 *
 * To create a message, you typically want to use the message-creation
 * tools in the Transaction class.  The Transaction class automatically
 * allocates space for any headers and fills them out, to avoid having
 * to copy the message's contents multiple times.
 */
class Message {
  FORBID_ACCIDENTAL_COPIES(Message);

 private:
  /** Rank of the message's source or destination (the one that isn't me). */
  int peer_;
  /** The channel associated with this message (-1 if it is a response). */
  int channel_;
  /** The transaction ID associated with the message. */
  int transaction_id_;
  /** All memory allocated to the message. */
  char *buffer_;
  /** Memory dedicated to the payload of the message. */
  size_t header_size_;
  /** Amount of memory allocated to the payload. */
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
    header_size_ = header_size_in;
    data_size_ = data_size_in;
  }

  size_t data_size() const {
    return data_size_;
  }
  size_t header_size() const {
    return header_size_;
  }
  size_t buffer_size() const {
    return header_size_ + data_size_;
  }
  char *data() const {
    return buffer_ + header_size_;
  }
  template<typename T>
  T *data_as() const {
    return reinterpret_cast<T*>(data());
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

/**
 * A Transaction is an agreed-upon communication event, which you may
 * think of as a "state machine".
 *
 * Synchronous message passing systems consider sending one message at
 * a time along with a tag.  For asynchronous message passing, a single
 * "transaction" might be a lot more complicated than just sending or
 * receiving a single message.  However, many of these transactions may be
 * in flight at the same time, and it becomes necessary to keep track of
 * each transaction separately because each one has different internal
 * state.  Instead of a tag, a "channel" is an agreed-upon number used
 * to start up new transactions, after which a dynamically-assigned
 * transaction number is used for the remainder of the transaction.
 *
 * To use Transactions, either use a BasicTransaction (which mimics
 * traditional synchronous message passing) or inherit from this class
 * and implement HandleMessage.  If you implement your own, you should
 * store any internal state in the class itself.  Keep in mind that
 * transactions are used from two contexts: one is created on the requesting
 * side by client code, and one is created on the Channel side
 * asynchronously.  The RPC system makes no assumptions about how you
 * manage your memory, so a transaction that is allocated on the
 * heap needs to know when to delete itself.
 *
 * A Transaction becomes registered with a transaction number on each
 * machine it is involved with.  Both machines keep track of vacant
 * transaction numbers and will recycle them, so unregistering a transaction
 * properly is very important.  The Done() functions are responsible for
 * freeing up a transaction number for recycling, and both sides must
 * make sure to agree upon it.
 *
 * Of all the difficulties in programming asynchronously, we make a major
 * guarantee: HandleMessage is only called from the network thread and
 * thus is only called once at a time.  Therefore, when calling all of
 * the functions within Transaction, if you are calling from within
 * the context of HandleMessage, you don't have to worry about another
 * response being received quickly.
 *
 * When calling from another thread, beware of receiving a response from the
 * other machine faster than you'd expect.
 */
class Transaction {
  FORBID_ACCIDENTAL_COPIES(Transaction);

 private:
  int channel_;
  struct PeerInfo {
    int peer;
    int channel;
    int transaction_id;
  };
  std::vector<PeerInfo> peers_;

 public:
  /** Create a message of a specified size, which you will later Send(). */
  Message *CreateMessage(int peer, size_t size);
  /** Send a message */
  void Send(Message *message);
  /** Unregister the transaction from all peers */
  void Done();
  /** Unregister the transaction from just this peer */
  void Done(int peer);

 public:
  Transaction() {
    channel_ = BIG_BAD_NUMBER;
  }
  virtual ~Transaction() { channel_ = BIG_BAD_NUMBER; }

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
   * @param message incoming message - keep this around as long as you like,
   *        but make sure to delete it when you are done
   */
  virtual void HandleMessage(Message *message) = 0;
};

/**
 * A Channel is the mechanism for listening for new messages and starting
 * new transactions.
 *
 * Of all parts of the "RPC" system, channels are the only
 * component that used for message routing that must be agreed on beforehand
 * because without this, establishing new transactions is not possible.
 * (All other message routing is done via established transactions).
 * A channel must have a predetermined number that is unique and the same
 * for all machines.  Do not use channels less than 100 (these are reserved
 * for the implementation).
 *
 * Re-using channels right after another might be a bad idea, so if you have
 * two barriers in a row, use different channels.  Fixing this bug correctly
 * is a possible future task, and probably related to the way that barriers
 * are defined :-)
 */
class Channel {
 public:
  Channel() {}
  virtual ~Channel() {}

  virtual Transaction *GetTransaction(Message *message) {
    FATAL("GetTransaction not implemented?");
  }
};

#endif

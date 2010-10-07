/** @file mailbox.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_MAILBOX_H
#define CORE_TABLE_MAILBOX_H

#include "core/table/distributed_table_message.h"

namespace core {
namespace table {

class Table;

class PointInbox {
  private:

    boost::condition_variable point_received_cond_;

    boost::mpi::communicator *comm_;

    std::vector<double> point_;

    bool point_handle_is_valid_;

    boost::mpi::request point_handle_;

    boost::shared_ptr<boost::thread> point_inbox_thread_;

    boost::mutex termination_mutex_;

    boost::condition_variable termination_cond_;

  public:

    boost::mutex &termination_mutex() {
      return termination_mutex_;
    }

    boost::condition_variable &termination_cond() {
      return termination_cond_;
    }

    void Detach() {
      point_inbox_thread_->detach();
    }

    PointInbox() {
      comm_ = NULL;
      point_handle_is_valid_ = false;
    }

    void Init(boost::mpi::communicator *comm_in) {

      // Set the communicator.
      comm_ = comm_in;

      // Start the point inbox thread.
      point_inbox_thread_ = boost::shared_ptr<boost::thread>(
                              new boost::thread(
                                boost::bind(
                                  &core::table::PointInbox::server,
                                  this)));
    }

    bool has_outstanding_point_messages() {
      return comm_->iprobe(
               boost::mpi::any_source,
               core::table::DistributedTableMessage::RECEIVE_POINT);
    }

    bool termination_signal_arrived() {
      return
        comm_->iprobe(
          boost::mpi::any_source,
          core::table::DistributedTableMessage::TERMINATE_POINT_INBOX);
    }

    bool time_to_quit() {
      return point_handle_is_valid_ == false &&
             has_outstanding_point_messages() == false &&
             termination_signal_arrived();
    }

    bool point_received() {
      return point_handle_is_valid_ && point_handle_.test();
    }

    void server() {
      while(this->time_to_quit()) {

        // Probe the message queue for the point request, and do an
        // asynchronous receive, if we can.
        if(point_handle_is_valid_ == false &&
            this->has_outstanding_point_messages()) {

          // Set the valid flag on and start receiving the point.
          point_handle_is_valid_ = true;
          point_handle_ =
            comm_->irecv(
              boost::mpi::any_source,
              core::table::DistributedTableMessage::REQUEST_POINT,
              point_);
        }

        // Check whether the request is done.
        if(this->point_received()) {

          // Wake up the thread waiting on the request. The woken up
          // thread turns off the validity flag after grabbing whch
          // process wants a point.
          point_received_cond_.notify_one();
        }
      }
    }
};

class PointRequestMessageInbox {
  private:

    boost::condition_variable point_request_message_received_cond_;

    boost::mpi::communicator *comm_;

    core::table::PointRequestMessage point_request_message_;

    bool point_request_message_is_valid_;

    boost::mpi::request point_request_message_handle_;

    boost::shared_ptr<boost::thread> point_request_message_inbox_thread_;

  public:

    void Detach() {
      point_request_message_inbox_thread_->detach();
    }

    void export_point_request_message(
      int *source_rank_out, int *point_id_out) const {

      *source_rank_out = point_request_message_.source_rank();
      *point_id_out = point_request_message_.point_id();
    }

    void invalidte_point_request_message() {
      point_request_message_is_valid_ = false;
    }

    void Init(boost::mpi::communicator *comm_in) {
      comm_ = comm_in;

      // Start the point request message inbox thread.
      point_request_message_inbox_thread_ =
        boost::shared_ptr<boost::thread>(
          new boost::thread(
            boost::bind(
              &core::table::PointRequestMessageInbox::server, this)));
    }

    PointRequestMessageInbox() {
      comm_ = NULL;
      point_request_message_is_valid_ = false;
    }

    void invalidate_point_request_message() {
      point_request_message_is_valid_ = false;
    }

    bool termination_signal_arrived() {
      return
        comm_->iprobe(
          boost::mpi::any_source,
          core::table::DistributedTableMessage::TERMINATE_POINT_REQUEST_MESSAGE_INBOX);
    }

    bool time_to_quit() {

      // It is time to quit when there are no valid messages to
      // handle, and the terminate signal is here.
      return point_request_message_is_valid_ == false &&
             has_outstanding_point_request_messages() == false &&
             termination_signal_arrived();
    }

    bool point_request_message_received() {
      return point_request_message_is_valid_ &&
             point_request_message_handle_.test();
    }

    bool has_outstanding_point_request_messages() {
      return comm_->iprobe(
               boost::mpi::any_source,
               core::table::DistributedTableMessage::REQUEST_POINT);
    }

    void server() {
      while(this->time_to_quit()) {

        // Probe the message queue for the point request, and do an
        // asynchronous receive, if we can.
        if(point_request_message_is_valid_ == false &&
            this->has_outstanding_point_request_messages()) {

          // Set the valid flag on and start receiving the message.
          point_request_message_is_valid_ = true;
          point_request_message_handle_ =
            comm_->irecv(
              boost::mpi::any_source,
              core::table::DistributedTableMessage::REQUEST_POINT,
              point_request_message_);
        }

        // Check whether the request is done.
        if(this->point_request_message_received()) {

          // Wake up the thread waiting on the request. This thread
          // turns off the validity flag after grabbing whch process
          // wants a point.
          point_request_message_received_cond_.notify_one();
        }

      } // end of the infinite server loop.

      // Kill the outbox handling the point request messages.
      comm_->isend(
        comm_->rank(),
        core::table::DistributedTableMessage::TERMINATE_POINT_REQUEST_MESSAGE_OUTBOX,
        0);
    }
};

class PointRequestMessageOutbox {
  private:

    core::table::PointRequestMessageInbox *inbox_;

    boost::mpi::communicator *comm_;

    core::table::Table *table_;

    boost::mutex mutex_;

    boost::condition_variable *point_request_message_received_cond_;

    std::vector<double> outgoing_point_;

    bool point_request_message_is_valid_;

    boost::mpi::request point_request_message_handle_;

    boost::shared_ptr<boost::thread> point_request_message_outbox_thread_;

  public:

    void Detach() {
      point_request_message_outbox_thread_->detach();
    }

    PointRequestMessageOutbox() {
      inbox_ = NULL;
      comm_ = NULL;
      table_ = NULL;
      point_request_message_is_valid_ = false;
    }

    void Init(
      boost::mpi::communicator *comm_in,
      core::table::Table *table_in,
      core::table::PointRequestMessageInbox *inbox_in,
      boost::condition_variable *point_request_message_received_cond_in) {

      comm_ = comm_in;
      table_ = table_in;
      inbox_ = inbox_in;
      point_request_message_received_cond_ =
        point_request_message_received_cond_in;

      // Start the point request message oubox thread.
      point_request_message_outbox_thread_ = boost::shared_ptr<boost::thread>(
          new boost::thread(
            boost::bind(
              &core::table::PointRequestMessageOutbox::server, this)));
    }

    bool termination_signal_arrived() {
      return comm_->iprobe(
               boost::mpi::any_source,
               core::table::DistributedTableMessage::TERMINATE_POINT_REQUEST_MESSAGE_OUTBOX);
    }

    bool time_to_quit() {

      // It is time to quit when there are no valid messages to
      // handle, and the terminate signal is here.
      return point_request_message_is_valid_ == false &&
             termination_signal_arrived();
    }

    void server() {
      while(this->time_to_quit()) {

        // Lock the mutex so that it can wait on the condition
        // variable.
        boost::unique_lock<boost::mutex> lock(mutex_);

        // If no message is available from the inbox, then go to
        // sleep.
        if(inbox_->point_request_message_received() == false) {
          point_request_message_received_cond_->wait(lock);
        }

        // If the server is free to send out points, then
        if(point_request_message_is_valid_ == false) {
          int source_rank;
          int point_id;
          inbox_->export_point_request_message(&source_rank, &point_id);

          // Invalid the message in the inbox.
          inbox_->invalidate_point_request_message();

          // Grab the point from the table.
          table_->get(point_id, &outgoing_point_);

          // Send back the point to the requester. Lock the outgoing
          // point so that it is not overwritten while doing the
          // transfer.
          point_request_message_is_valid_ = true;
          point_request_message_handle_ =
            comm_->isend(
              source_rank,
              core::table::DistributedTableMessage::RECEIVE_POINT,
              outgoing_point_);
        }

        // Otherwise, check whether the transfer is done and unlock.
        else if(point_request_message_handle_.test()) {
          point_request_message_is_valid_ = false;
        }

      } // end of the infinite server loop.
    }
};
};
};

#endif

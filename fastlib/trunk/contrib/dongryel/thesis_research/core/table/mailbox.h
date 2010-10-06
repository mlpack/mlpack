/** @file point_request_message.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_MAILBOX_H
#define CORE_TABLE_MAILBOX_H

#include "core/table/distributed_table_message.h"

namespace core {
namespace table {

class Table;

class PointRequestMessageInbox {
  private:
    boost::condition_variable point_request_message_received_cond_;

    boost::mpi::communicator *comm_;

    bool point_request_message_is_valid_;

    core::table::PointRequestMessage point_request_message_;

    boost::mpi::request point_request_message_handle_;

  public:

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
    }

    PointRequestMessageInbox() {
      comm_ = NULL;
      point_request_message_is_valid_ = false;
    }

    bool termination_signal_arrived() {
      return comm_->iprobe(
               boost::mpi::any_source,
               core::table::DistributedTableMessage::TERMINATE_SERVER);
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
      while(true) {

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

        // Check whether it is time to quit.
        if(this->time_to_quit()) {
          break;
        }
      } // end of the infinite server loop.
    }
};

class PointRequestMessageOutbox {
  private:

    core::table::PointRequestMessageInbox *inbox_;

    core::table::Table *table_;

    boost::mutex mutex_;

    boost::condition_variable *point_request_message_received_cond_;

  public:
    void Init(
      core::table::Table *table_in,
      core::table::PointRequestMessageInbox *inbox_in,
      boost::condition_variable *point_request_message_received_cond_in) {
      table_ = table_in;
      inbox_ = inbox_in;
      point_request_message_received_cond_ =
        point_request_message_received_cond_in;
    }

    void server() {
      while(true) {

        boost::unique_lock<boost::mutex> lock(mutex_);

        // If no message is available from the inbox, then go to
        // sleep.
        if(inbox_->point_request_message_received() == false) {
          point_request_message_received_cond_->wait();
        }

        // Grab what we want from the inbox, and invalidate the
        // message in the inbox.
        int source_rank;
        int point_id;
        inbox_->export_point_request_message(&source_rank, &point_id);
        inbox_->invalidate_point_request_message();

        // Send the point to the requestor.
      }
    }
};

class Mailbox {
  public:

    boost::mutex termination_mutex_;

    boost::condition_variable termination_cond_;

    boost::mutex mutex_;

    boost::condition_variable point_ready_cond_;

    boost::mpi::communicator *communicator_;

    std::pair <
    boost::mpi::request,
          core::table::PointRequestMessage > incoming_request_;

    boost::mpi::request outgoing_request_;

    std::pair< boost::mpi::request, bool> incoming_receive_request_;

    int incoming_receive_source_;

    std::vector<double> incoming_point_;

    std::vector<double> outgoing_point_;

  public:

    bool is_done() {

      bool first = (communicator_->iprobe(
                      boost::mpi::any_source,
                      core::table::DistributedTableMessage::TERMINATE_SERVER));
      bool second = first &&
                    (!communicator_->iprobe(
                       boost::mpi::any_source,
                       core::table::DistributedTableMessage::REQUEST_POINT));
      bool third = second &&
                   (!communicator_->iprobe(
                      boost::mpi::any_source,
                      core::table::DistributedTableMessage::RECEIVE_POINT));
      bool fourth = (
                      third && incoming_request_.second.is_valid() == false &&
                      incoming_receive_request_.second == false);
      if(fourth) {
        try {
          bool outgoing_request_done = outgoing_request_.test();
          fourth = fourth && outgoing_request_done;
        }
        catch(boost::mpi::exception &e) {
        }
      }
      return fourth;
    }

    void set_communicator(boost::mpi::communicator *comm_in) {
      communicator_ = comm_in;
    }

    Mailbox() {
      communicator_ = NULL;
      incoming_receive_request_.second = false;
    }
};
};
};

#endif

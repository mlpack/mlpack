/** @file mailbox.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_MAILBOX_H
#define CORE_TABLE_MAILBOX_H

#include "boost/mpi/communicator.hpp"
#include "core/table/distributed_table_message.h"

namespace core {
namespace table {

class Table;

class PointInbox {
  private:

    boost::mutex point_received_mutex_;

    boost::condition_variable point_received_cond_;

    boost::mpi::communicator *comm_;

    std::vector<double> point_;

    bool point_handle_is_valid_;

    bool do_test_;

    boost::mpi::request point_handle_;

    boost::shared_ptr<boost::thread> point_inbox_thread_;

    boost::mutex termination_mutex_;

    boost::condition_variable termination_cond_;

  public:

    void invalidate_point() {
      point_handle_is_valid_ = false;
      do_test_ = true;
    }

    const std::vector<double> &point() const {
      return point_;
    }

    void wait(boost::unique_lock<boost::mutex> &lock_in) {
      point_received_cond_.wait(lock_in);
    }

    boost::mutex &point_received_mutex() {
      return point_received_mutex_;
    }

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
      do_test_ = true;
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

      comm_->barrier();
    }

    bool has_outstanding_point_messages() {
      int flag;
      MPI_Status status;
      MPI_Iprobe(
        MPI_ANY_SOURCE,
        core::table::DistributedTableMessage::RECEIVE_POINT,
        comm_->operator MPI_Comm(), &flag, &status);
      return flag;
    }

    bool termination_signal_arrived() {
      int flag;
      MPI_Status status;
      MPI_Iprobe(
        MPI_ANY_SOURCE,
        core::table::DistributedTableMessage::TERMINATE_POINT_INBOX,
        comm_->operator MPI_Comm(), &flag, &status);
      return flag;
    }

    bool time_to_quit() {
      return point_handle_is_valid_ == false &&
             (! has_outstanding_point_messages()) &&
             termination_signal_arrived();
    }

    bool point_received() {
      return point_handle_is_valid_ && point_handle_.test();
    }

    void server() {

      do {

        // Probe the message queue for the point request, and do an
        // asynchronous receive, if we can.
        if(point_handle_is_valid_ == false &&
            this->has_outstanding_point_messages()) {

          // Set the valid flag on and start receiving the point.
          printf("Starting to receive a point.\n");
          point_handle_is_valid_ = true;
          point_handle_ =
            comm_->irecv(
              boost::mpi::any_source,
              core::table::DistributedTableMessage::RECEIVE_POINT,
              point_);
        }

        // Check whether the request is done.
        if(do_test_ && this->point_received()) {

          // Wake up the thread waiting on the request. The woken up
          // thread turns off the validity flag after grabbing whch
          // process wants a point.
          printf("Waking the main thread up!\n");
          do_test_ = false;
          point_received_cond_.notify_one();
        }
      }
      while(this->time_to_quit() == false);     // end of the server loop.

      printf("Point inbox for Process %d is quitting.\n", comm_->rank());
      termination_cond_.notify_one();
    }
};

class PointRequestMessageBox {
  private:

    core::table::Table *owned_table_;

    boost::condition_variable point_request_message_box_quitting_;

    boost::mpi::communicator *comm_;

    core::table::PointRequestMessage point_request_message_;

    bool point_request_message_is_valid_;

    boost::mpi::request point_request_message_handle_;

    bool point_request_message_sent_is_valid_;

    boost::mpi::request point_request_message_sent_handle_;

    std::vector<double> outgoing_point_;

    boost::shared_ptr<boost::thread> point_request_message_box_thread_;

    boost::mutex mutex_;

  public:

    boost::mutex &mutex() {
      return mutex_;
    }

    boost::condition_variable &point_request_message_box_quitting() {
      return point_request_message_box_quitting_;
    }

    void Detach() {
      point_request_message_box_thread_->detach();
    }

    void export_point_request_message(
      int *source_rank_out, int *point_id_out) const {

      *source_rank_out = point_request_message_.source_rank();
      *point_id_out = point_request_message_.point_id();
    }

    void Init(
      boost::mpi::communicator *comm_in,
      core::table::Table *owned_table_in) {

      comm_ = comm_in;
      owned_table_ = owned_table_in;

      // Start the point request message inbox thread.
      point_request_message_box_thread_ =
        boost::shared_ptr<boost::thread>(
          new boost::thread(
            boost::bind(
              &core::table::PointRequestMessageBox::server, this)));

      comm_->barrier();
    }

    PointRequestMessageBox() {
      comm_ = NULL;
      point_request_message_is_valid_ = false;
      point_request_message_sent_is_valid_ = false;
    }

    bool termination_signal_arrived() {
      int flag;
      MPI_Status status;
      MPI_Iprobe(
        MPI_ANY_SOURCE,
        core::table::DistributedTableMessage::TERMINATE_POINT_REQUEST_MESSAGE_BOX,
        comm_->operator MPI_Comm(), &flag, &status);
      return flag;
    }

    bool time_to_quit() {

      // It is time to quit when there are no valid messages to
      // handle, and the terminate signal is here.
      return point_request_message_is_valid_ == false &&
             point_request_message_sent_is_valid_ == false &&
             (! has_outstanding_point_request_messages()) &&
             termination_signal_arrived();
    }

    bool point_request_message_received() {
      return point_request_message_is_valid_ &&
             point_request_message_handle_.test();
    }

    bool point_request_message_sent() {
      return point_request_message_sent_is_valid_ &&
             point_request_message_sent_handle_.test();
    }

    bool has_outstanding_point_request_messages() {
      int flag;
      MPI_Status status;
      MPI_Iprobe(
        MPI_ANY_SOURCE,
        core::table::DistributedTableMessage::REQUEST_POINT,
        comm_->operator MPI_Comm(), &flag, &status);
      return flag;
    }

    void server() {

      do {

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
          printf("Process %d is receiving a request message.\n",
                 comm_->rank());
        }

        // Check whether the incoming receive is done. If so, send out
        // the requested point asynchronously given that the outgoing
        // buffer is available.
        if(point_request_message_sent_is_valid_ == false &&
            this->point_request_message_received()) {
          int source_rank = point_request_message_.source_rank();
          int point_id = point_request_message_.point_id();

          owned_table_->get(point_id, &outgoing_point_);

          point_request_message_sent_is_valid_ = true;
          point_request_message_sent_handle_ = comm_->isend(
                                                 source_rank,
                                                 core::table::DistributedTableMessage::RECEIVE_POINT,
                                                 outgoing_point_);

          point_request_message_is_valid_ = false;
        }
        if(this->point_request_message_sent()) {

          // Turn off the boolean flag so that the buffer can be
          // reused.
          point_request_message_sent_is_valid_ = false;
        }

      }
      while(this->time_to_quit() == false) ;
      // end of the infinite server loop.

      printf("Point request message inbox for Process %d is quitting.\n",
             comm_->rank());
      point_request_message_box_quitting_.notify_one();
    }
};
};
};

#endif

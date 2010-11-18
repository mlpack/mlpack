/** @file mailbox.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_MAILBOX_H
#define CORE_TABLE_MAILBOX_H

#include <boost/interprocess/offset_ptr.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/mpi/communicator.hpp>
#include "core/table/distributed_table_message.h"
#include "core/table/point_request_message.h"

namespace core {
namespace table {

class TableInbox {
  private:

    bool received_point_is_locked_;

    boost::interprocess::offset_ptr<double> received_point_;

    int n_attributes_;

    int num_time_to_quit_signals_;

  public:

    const double *get_point(int source_rank, int point_id) const {
      return received_point_.get();
    }

    void UnlockPoint() {
      received_point_is_locked_ = false;
    }

    TableInbox() {
      received_point_is_locked_ = false;
      received_point_ = NULL;
      n_attributes_ = -1;
      num_time_to_quit_signals_ = 0;
    }

    ~TableInbox() {
      if(received_point_ != NULL) {
        core::table::global_m_file_->DestroyPtr(received_point_.get());
      }
    }

    void Init(int num_dimensions_in) {

      received_point_is_locked_ = false;
      received_point_ = core::table::global_m_file_->ConstructArray<double>(
                          num_dimensions_in);
      n_attributes_ = num_dimensions_in;
    }

    void Run(
      boost::mpi::intercommunicator &inbox_to_outbox_comm_in,
      boost::mpi::intercommunicator &inbox_to_computation_comm_in) {

      do {

        // Probe the message queue for the point request, and do a
        // receive.
        if(received_point_is_locked_ == false &&
            inbox_to_outbox_comm_in.iprobe(
              boost::mpi::any_source,
              core::table::DistributedTableMessage::
              RECEIVE_POINT_FROM_TABLE_OUTBOX)) {

          boost::mpi::request recv_request =
            inbox_to_outbox_comm_in.irecv(
              boost::mpi::any_source,
              core::table::DistributedTableMessage::
              RECEIVE_POINT_FROM_TABLE_OUTBOX,
              received_point_.get(), n_attributes_);
          recv_request.wait();

          // The point remains valid until it is received from the
          // other end.
          received_point_is_locked_ = true;

          // Notify the computation process.
          inbox_to_computation_comm_in.send(
            inbox_to_computation_comm_in.rank(),
            core::table::DistributedTableMessage::
            RECEIVE_POINT_FROM_TABLE_INBOX, 0);
        }

        // Check if any quit signal is here.
        if(inbox_to_computation_comm_in.iprobe(
              boost::mpi::any_source,
              core::table::DistributedTableMessage::
              TERMINATE_TABLE_INBOX)) {
          int dummy;
          boost::mpi::request recv_request =
            inbox_to_computation_comm_in.irecv(
              boost::mpi::any_source,
              core::table::DistributedTableMessage::
              TERMINATE_TABLE_INBOX, dummy);
          recv_request.wait();
          num_time_to_quit_signals_++;
        }
      }
      while(
        num_time_to_quit_signals_ <
        inbox_to_computation_comm_in.remote_size());

      printf("Table inbox for Process %d is quitting.\n",
             inbox_to_outbox_comm_in.local_rank());
    }
};

template<typename TableType>
class TableOutbox {
  private:

    core::table::PointRequestMessage point_request_message_;

    boost::interprocess::offset_ptr<TableType> *owned_table_;

    bool time_to_quit_;

    int num_time_to_quit_signals_;

  public:

    void Init(
      boost::interprocess::offset_ptr<TableType> &owned_table_in) {

      owned_table_ = &owned_table_in;
      num_time_to_quit_signals_ = 0;
    }

    void Run(
      boost::mpi::intercommunicator &outbox_to_inbox_comm_in,
      boost::mpi::intercommunicator &outbox_to_computation_comm_in) {

      do {

        // Probe the message queue for the point request, and receive
        // the message to find out whom to send stuffs to.
        if(outbox_to_computation_comm_in.iprobe(
              boost::mpi::any_source,
              core::table::DistributedTableMessage::
              REQUEST_POINT_FROM_TABLE_OUTBOX)) {

          // Receive from the computation group.
          boost::mpi::request recv_request =
            outbox_to_computation_comm_in.irecv(
              boost::mpi::any_source,
              core::table::DistributedTableMessage::
              REQUEST_POINT_FROM_TABLE_OUTBOX,
              point_request_message_);
          recv_request.wait();

          // Send it to its inbox.
          outbox_to_inbox_comm_in.send(
            point_request_message_.source_rank(),
            core::table::DistributedTableMessage::
            RECEIVE_POINT_FROM_TABLE_OUTBOX,
            (*owned_table_)->GetColumnPtr(
              point_request_message_.point_id()),
            (*owned_table_)->n_attributes());
        }

        // Check if any quit signal is here.
        if(outbox_to_computation_comm_in.iprobe(
              boost::mpi::any_source,
              core::table::DistributedTableMessage::
              TERMINATE_TABLE_OUTBOX)) {
          int dummy;
          boost::mpi::request recv_request =
            outbox_to_computation_comm_in.irecv(
              boost::mpi::any_source,
              core::table::DistributedTableMessage::
              TERMINATE_TABLE_OUTBOX, dummy);
          recv_request.wait();
          num_time_to_quit_signals_++;
        }
      }
      while(
        num_time_to_quit_signals_ <
        outbox_to_computation_comm_in.remote_size() ||
        outbox_to_computation_comm_in.iprobe(
          boost::mpi::any_source,
          core::table::DistributedTableMessage::
          REQUEST_POINT_FROM_TABLE_OUTBOX));

      // end of the infinite server loop.
      printf("Table outbox for Process %d is quitting.\n",
             outbox_to_inbox_comm_in.local_rank());
    }
};
};
};

#endif

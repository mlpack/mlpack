/** @file mailbox.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_MAILBOX_H
#define CORE_TABLE_MAILBOX_H

#include <boost/interprocess/offset_ptr.hpp>
#include <boost/mpi/communicator.hpp>
#include "core/table/distributed_table_message.h"

namespace core {
namespace table {

class TableInbox {
  private:

    bool received_point_is_valid_;

    boost::interprocess::offset_ptr<double> received_point_;

    int n_attributes_;

    bool time_to_quit_;

  public:

    TableInbox() {
      received_point_is_valid_ = false;
      received_point_ = NULL;
      n_attributes_ = -1;
      time_to_quit_ = false;
    }

    ~TableInbox() {
      if(received_point_ != NULL) {
        core::table::global_m_file_->DestroyPtr(received_point_.get());
      }
    }

    void Init(int num_dimensions_in) {

      received_point_is_valid_ = false;
      received_point_ = core::table::global_m_file_->ConstructArray<double>(
                          num_dimensions_in);
      n_attributes_ = num_dimensions_in;
    }

    void Run(
      boost::mpi::communicator &table_outbox_group_comm_in,
      boost::mpi::communicator &table_inbox_group_comm_in,
      boost::mpi::communicator &computation_group_comm_in) {

      do {

        // Probe the message queue for the point request, and do a
        // receive.
        if(received_point_is_valid_ == false &&
            table_outbox_group_comm_in.iprobe(
              boost::mpi::any_source,
              core::table::DistributedTableMessage::
              RECEIVE_POINT_FROM_TABLE_OUTBOX)) {

          table_outbox_group_comm_in.recv(
            boost::mpi::any_source,
            core::table::DistributedTableMessage::
            RECEIVE_POINT_FROM_TABLE_OUTBOX,
            received_point_.get(), n_attributes_);

          // The point remains valid until it is received from the
          // other end.
          received_point_is_valid_ = true;
        }
      }
      while(time_to_quit_ == false);

      printf("Table inbox for Process %d is quitting.\n",
             table_inbox_group_comm_in.rank());
    }
};

template<typename TableType>
class TableOutbox {
  private:

    boost::interprocess::offset_ptr<TableType> *owned_table_;

  public:

    void Init(
      boost::interprocess::offset_ptr<TableType> &owned_table_in) {

    }

    void Run(
      boost::mpi::communicator &table_outbox_group_comm_in,
      boost::mpi::communicator &table_inbox_group_comm_in,
      boost::mpi::communicator &computation_group_comm_in) {

      do {

        // Probe the message queue for the point request, and receive
        // the message to find out whom to send stuffs to.


      }
      while(true) ;

      // end of the infinite server loop.
      printf("Table outbox for Process %d is quitting.\n",
             table_outbox_group_comm_in.rank());
    }
};
};
};

#endif

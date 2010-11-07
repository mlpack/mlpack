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

template<typename TableType>
class TableInbox {
  private:
    boost::mpi::communicator *global_comm_;

    boost::mpi::communicator *table_outbox_group_comm_;

    boost::mpi::communicator *table_inbox_group_comm_;

    bool received_point_is_valid_;

    double *received_point_;

    TableType *owned_table_;

  public:

    void Init(
      TableType *owned_table_in,
      boost::mpi::communicator *global_comm_in,
      boost::mpi::communicator *table_outbox_group_comm_in,
      boost::mpi::communicator *table_inbox_group_comm_in) {

      global_comm_ = global_comm_in;
      table_outbox_group_comm_ = table_outbox_group_comm_in;
      table_inbox_group_comm_ = table_inbox_group_comm_in;

      received_pint_is_valid_ = false;
      received_point_ = new double[owned_table_in->n_attributes()];
      owned_table_ = owned_table_in;
    }

    void Run() {

      do {

        // Probe the message queue for the point request, and do a
        // receive.
        if(received_point_is_valid_ == false &&
            table_outbox_group_comm_->iprobe(
              boost::mpi::any_source,
              core::table::DistributedTableMessage::RECEIVE_POINT)) {

          table_outbox_group_comm_->recv(
            boost::mpi::any_source,
            core::table::DistributedTableMessage::RECEIVE_POINT,
            received_point_,
            owned_table_->n_attributes());
        }

      }
      while(this->time_to_quit() == false);     // end of the server loop.

      printf("Table inbox for Process %d is quitting.\n", comm_->rank());
    }
};

template<typename TableType>
class TableOutbox {
  private:

    boost::mpi::communicator *global_comm_;

    boost::mpi::communicator *table_outbox_group_comm_;

    boost::mpi::communicator *table_inbox_group_comm_;

    TableType *owned_table_;

  public:

    void Run() {

      do {

        // Probe the message queue for the point request, and receive
        // the message to find out whom to send stuffs to.


      }
      while(this->time_to_quit() == false) ;

      // end of the infinite server loop.
      printf("Point request message inbox for Process %d is quitting.\n",
             comm_->rank());
    }
};
};
};

#endif

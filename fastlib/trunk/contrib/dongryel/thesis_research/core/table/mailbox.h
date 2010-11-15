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

    int n_attributes_;

  public:

    TableInbox() {
      global_comm_ = NULL;
      table_outbox_group_comm_ = NULL;
      table_inbox_group_comm_ = NULL;
      received_point_is_valid_ = false;
      received_point_ = NULL;
      n_attributes_ = -1;
    }

    ~TableInbox() {
      if(received_point_ != NULL) {
        core::table::global_m_file_->DestroyPtr(received_point_);
      }
    }

    void Init(
      int num_dimensions_in,
      boost::mpi::communicator *global_comm_in,
      boost::mpi::communicator *table_outbox_group_comm_in,
      boost::mpi::communicator *table_inbox_group_comm_in) {

      global_comm_ = global_comm_in;
      table_outbox_group_comm_ = table_outbox_group_comm_in;
      table_inbox_group_comm_ = table_inbox_group_comm_in;

      received_point_is_valid_ = false;
      received_point_ = core::table::global_m_file_->ConstructArray<double>(
                          num_dimensions_in);
      n_attributes_ = num_dimensions_in;
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
            n_attributes_);
        }

      }
      while(
        global_comm_->iprobe(
          boost::mpi::any_source,
          core::table::DistributedTableMessage::TERMINATE_TABLE_INBOX));

      // Receive the message.
      int dummy;
      global_comm_->recv(
        boost::mpi::any_source,
        core::table::DistributedTableMessage::TERMINATE_TABLE_INBOX,
        dummy);
      printf("Table inbox for Process %d is quitting.\n",
             table_inbox_group_comm_->rank());
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

    void Init(
      TableType *owned_table_in,
      boost::mpi::communicator *global_comm_in,
      boost::mpi::communicator *table_outbox_group_comm_in,
      boost::mpi::communicator *table_inbox_group_comm_in) {

      global_comm_ = global_comm_in;
      table_outbox_group_comm_ = table_outbox_group_comm_in;
      table_inbox_group_comm_ = table_inbox_group_comm_in;
    }

    void Run() {

      do {

        // Probe the message queue for the point request, and receive
        // the message to find out whom to send stuffs to.


      }
      while(true) ;

      // end of the infinite server loop.
      printf("Table outbox for Process %d is quitting.\n",
             table_outbox_group_comm_->rank());
    }
};
};
};

#endif

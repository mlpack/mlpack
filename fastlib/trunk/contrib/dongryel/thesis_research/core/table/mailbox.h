/** @file point_request_message.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_MAILBOX_H
#define CORE_TABLE_MAILBOX_H

#include "core/table/distributed_table_message.h"

namespace core {
namespace table {
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

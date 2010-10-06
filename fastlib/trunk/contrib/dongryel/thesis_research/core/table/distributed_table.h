/** @file distributed_table.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_DISTRIBUTED_TABLE_H
#define CORE_TABLE_DISTRIBUTED_TABLE_H

#include <armadillo>
#include "boost/mpi.hpp"
#include "boost/mpi/collectives.hpp"
#include "boost/thread.hpp"
#include "boost/serialization/string.hpp"
#include "core/table/table.h"
#include "core/table/distributed_table_message.h"
#include "core/table/point_request_message.h"
#include "core/table/mailbox.h"

namespace core {
namespace table {
class DistributedTable: public boost::noncopyable {

    typedef core::tree::GeneralBinarySpaceTree < core::tree::BallBound <
    core::table::DensePoint > > TreeType;

  private:

    core::table::Table *owned_table_;

    std::vector<int> local_n_entries_;

    TreeType *global_tree_;

    boost::mpi::communicator *comm_;

    boost::shared_ptr<boost::thread> table_thread_;

    core::table::Mailbox mailbox_;

  public:

    int rank() const {
      return comm_->rank();
    }

    bool IsIndexed() const {
      return global_tree_ != NULL;
    }

    DistributedTable() {
      comm_ = NULL;
      owned_table_ = NULL;
      global_tree_ = NULL;
    }

    ~DistributedTable() {

      // Put a barrier so that the distributed tables get destructed
      // when all of the requests are fulfilled for all of the
      // distributed processes.
      comm_->barrier();

      // Terminate the server.
      comm_->isend(
        comm_->rank(),
        core::table::DistributedTableMessage::TERMINATE_SERVER, 0);
      boost::unique_lock<boost::mutex> lock(mailbox_.termination_mutex_);
      mailbox_.termination_cond_.wait(lock);
      comm_->barrier();

      if(owned_table_ != NULL) {
        delete owned_table_;
        owned_table_ = NULL;
      }
      if(global_tree_ != NULL) {
        delete global_tree_;
        global_tree_ = NULL;
      }
    }

    const TreeType::BoundType &get_node_bound(TreeType *node) const {
      return node->bound();
    }

    TreeType::BoundType &get_node_bound(TreeType *node) {
      return node->bound();
    }

    TreeType *get_node_left_child(TreeType *node) {
      return node->left();
    }

    TreeType *get_node_right_child(TreeType *node) {
      return node->right();
    }

    bool node_is_leaf(TreeType *node) const {
      return node->is_leaf();
    }

    core::tree::AbstractStatistic *&get_node_stat(TreeType *node) {
      return node->stat();
    }

    int get_node_count(TreeType *node) const {
      return node->count();
    }

    TreeType *get_tree() {
      return global_tree_;
    }

    int n_attributes() const {
      return owned_table_->n_attributes();
    }

    int local_n_entries(int rank_in) const {
      return local_n_entries_[rank_in];
    }

    int local_n_entries() const {
      return owned_table_->n_entries();
    }

    void Init(
      const std::string &file_name,
      boost::mpi::communicator *communicator_in) {

      comm_ = communicator_in;
      owned_table_ = new core::table::Table();
      owned_table_->Init(file_name);
      mailbox_.set_communicator(communicator_in);

      // Allocate the vector for storing the number of entries for all
      // the tables in the world, and do an all-gather operation to
      // find out all the sizes.
      boost::mpi::all_gather(
        *comm_, owned_table_->n_entries(), local_n_entries_);

      // Start the server for giving out points.
      table_thread_ = boost::shared_ptr<boost::thread>(
                        new boost::thread(
                          boost::bind(
                            &core::table::DistributedTable::server,
                            this)));

      // Put a barrier to ensure that every process has started up the
      // server.
      comm_->barrier();

      // Detach the server thread for each distributed process.
      table_thread_->detach();
    }

    void Save(const std::string &file_name) const {

    }

    void IndexData(
      const core::metric_kernels::AbstractMetric &metric_in, int leaf_size) {

      // We need to build the top tree first.
      if(comm_->rank() == 0) {
      }
    }

    void server() {

      while(true) {

        // Probe the message queue for the point request, and do an
        // asynchronous receive first and buffer the receive.
        if(mailbox_.incoming_request_.second.is_valid() == false &&
            comm_->iprobe(
              boost::mpi::any_source,
              core::table::DistributedTableMessage::REQUEST_POINT)) {

          mailbox_.incoming_request_.second.set_valid();
          mailbox_.incoming_request_.first =
            comm_->irecv(
              boost::mpi::any_source,
              core::table::DistributedTableMessage::REQUEST_POINT,
              mailbox_.incoming_request_.second);
        }

        // See if the current outgoing request is done. If the current
        // incoming request is finished transferring, then send out
        // the new point.
        bool outgoing_request_done = false;
        try {
          outgoing_request_done = mailbox_.outgoing_request_.test();
        }
        catch(boost::mpi::exception &e) {
          outgoing_request_done = false;
        }
        if(outgoing_request_done &&
            mailbox_.incoming_request_.second.is_valid()) {

          bool incoming_request_done = false;
          try {
            incoming_request_done = mailbox_.incoming_request_.first.test();
          }
          catch(boost::mpi::exception &e) {
            incoming_request_done = false;
          }

          if(incoming_request_done) {
            // Get the reference to the incoming request to be
            // fulfilled.
            core::table::PointRequestMessage &to_be_flushed =
              mailbox_.incoming_request_.second;

            // Copy the point out.
            owned_table_->get(
              to_be_flushed.point_id(),
              &mailbox_.outgoing_point_);

            // Send back the point to the requester.
            mailbox_.outgoing_request_ =
              comm_->isend(
                to_be_flushed.source_rank(),
                core::table::DistributedTableMessage::RECEIVE_POINT,
                mailbox_.outgoing_point_);

            // This mail slot is free.
            to_be_flushed.Reset();
          }
        }

        // Check if any of the requested point from this processor is
        // ready to be received.
        if(mailbox_.incoming_receive_request_.second) {
          bool incoming_receive_request_done = false;
          try {
            incoming_receive_request_done =
              mailbox_.incoming_receive_request_.first.test();
          }
          catch(boost::mpi::exception &e) {
          }
          if(incoming_receive_request_done) {
            mailbox_.point_ready_cond_.notify_one();
          }
        }
        if(mailbox_.incoming_receive_request_.second == false &&
            comm_->iprobe(
              boost::mpi::any_source,
              core::table::DistributedTableMessage::RECEIVE_POINT)) {
          mailbox_.incoming_receive_request_.first =
            comm_->irecv(mailbox_.incoming_receive_source_,
                         core::table::DistributedTableMessage::RECEIVE_POINT,
                         mailbox_.incoming_point_);
          mailbox_.incoming_receive_request_.second = true;
        }

        // If the main thread has given a termination signal, and we
        // are done with everything, then the server thread exits.
        if(mailbox_.is_done()) {
          break;
        }

      } // end of the loop for handling buffered messages.

      mailbox_.termination_cond_.notify_one();
    }

    void get(
      int requested_rank, int point_id,
      core::table::DensePoint *entry) {

      // If owned by the process, just return the point. Otherwise, we
      // need to send an MPI request to the process holding the
      // required resource.
      if(comm_->rank() == requested_rank) {
        owned_table_->get(point_id, entry);
      }
      else {

        // The point request message.
        core::table::PointRequestMessage point_request_message(
          comm_->rank(), point_id);

        // Inform the source processor that this processor needs data!
        mailbox_.incoming_receive_source_ = requested_rank;
        comm_->isend(
          requested_rank,
          core::table::DistributedTableMessage::REQUEST_POINT,
          point_request_message);

        // Do a conditional wait until the point is ready.
        boost::unique_lock<boost::mutex> lock(mailbox_.mutex_);
        mailbox_.point_ready_cond_.wait(lock);

        // If we are here, then the point is ready. Copy the point.
        entry->Init(mailbox_.incoming_point_);

        // Signal that we are done copying out the point.
        mailbox_.incoming_receive_request_.second = false;
      }
    }

    void PrintTree() const {
      global_tree_->Print();
    }
};
};
};

#endif

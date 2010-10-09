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

    //core::table::PointInbox point_inbox_;

    core::table::PointRequestMessageInbox point_request_message_inbox_;

    //core::table::PointRequestMessageOutbox point_request_message_outbox_;

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

      // Terminate the point inbox.
      //comm_->isend(
      //comm_->rank(),
      //core::table::DistributedTableMessage::TERMINATE_POINT_INBOX, 0);

      // Terminate the point request message inbox.
      comm_->isend(
        comm_->rank(),
        core::table::DistributedTableMessage::TERMINATE_POINT_REQUEST_MESSAGE_INBOX, 0);
      boost::unique_lock<boost::mutex> lock(point_request_message_inbox_.mutex());
      point_request_message_inbox_.point_request_message_inbox_quitting().wait(lock);

      // Wait until the point inbox is terminated.
      //{
      //boost::unique_lock<boost::mutex> lock(point_inbox_.termination_mutex());
      //point_inbox_.termination_cond().wait(lock);
      //}

      // Put a barrier so that all processes are ready to destroy each
      // of their own tables and trees.
      comm_->barrier();

      // Delete the table.
      if(owned_table_ != NULL) {
        delete owned_table_;
        owned_table_ = NULL;
      }

      // Delete the tree.
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

      // Set the communicator and read the table.
      comm_ = communicator_in;
      owned_table_ = new core::table::Table();
      owned_table_->Init(file_name);

      // Allocate the vector for storing the number of entries for all
      // the tables in the world, and do an all-gather operation to
      // find out all the sizes.
      boost::mpi::all_gather(
        *comm_, owned_table_->n_entries(), local_n_entries_);

      // Initialize the mail boxes.
      //point_inbox_.Init(comm_);
      point_request_message_inbox_.Init(comm_);
      //point_request_message_outbox_.Init(
      //comm_, owned_table_, &point_request_message_inbox_);

      // Detach the server threads for each distributed process.
      //point_inbox_.Detach();
      point_request_message_inbox_.Detach();
      //point_request_message_outbox_.Detach();

      // Put a barrier to ensure that every process has started up the
      // mailboxes.
      comm_->barrier();
    }

    void Save(const std::string &file_name) const {

    }

    void IndexData(
      const core::metric_kernels::AbstractMetric &metric_in, int leaf_size) {

      // We need to build the top tree first.
      if(comm_->rank() == 0) {
      }
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
        comm_->isend(
          requested_rank,
          core::table::DistributedTableMessage::REQUEST_POINT,
          point_request_message);

        // Do a conditional wait until the point is ready.
        //boost::unique_lock<boost::mutex> lock(
        //point_inbox_.point_received_mutex());
        //point_inbox_.wait(lock);

        // If we are here, then the point is ready. Copy the point.
        //entry->Init(point_inbox_.point());

        // Signal that we are done copying out the point.
        //point_inbox_.invalidate_point();
      }
    }

    void PrintTree() const {
      global_tree_->Print();
    }
};
};
};

#endif

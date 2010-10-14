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

    /** @brief The mutex for each MPI call.
     */
    boost::mutex mpi_mutex_;

    core::table::Table *owned_table_;

    std::vector<int> local_n_entries_;

    TreeType *global_tree_;

    TreeType *global_tree_in_array_form_;

    boost::mpi::communicator *comm_;

    core::table::PointInbox point_inbox_;

    core::table::PointRequestMessageBox point_request_message_box_;

  public:

    void BroadcastTree_(TreeType *node) {
      if(! node->is_leaf()) {
        BroadcastTree_(node ->left());
      }

      // Broadcast the node.
      if(node != NULL) {
        mpi_mutex_.lock();
        boost::mpi::broadcast(* comm_, node, 0);
        mpi_mutex_.unlock();
      }

      if(! node->is_leaf()) {
        BroadcastTree_(node->right());
      }
    }

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
      global_tree_in_array_form_ = NULL;
    }

    ~DistributedTable() {

      // Put a barrier so that the distributed tables get destructed
      // when all of the requests are fulfilled for all of the
      // distributed processes.
      comm_->barrier();

      // Terminate the point request message box.
      {
        {
          boost::unique_lock<boost::mutex> lock_in(mpi_mutex_);
          comm_->isend(
            comm_->rank(),
            core::table::DistributedTableMessage::TERMINATE_POINT_REQUEST_MESSAGE_BOX, 0);
        }
        point_request_message_box_.Join();
      }

      // Wait until the point inbox is destroyed.
      {
        {
          boost::unique_lock<boost::mutex> lock_in(mpi_mutex_);
          comm_->isend(
            comm_->rank(),
            core::table::DistributedTableMessage::TERMINATE_POINT_INBOX, 0);
        }
        point_inbox_.Join();
      }

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
      if(global_tree_in_array_form_ != NULL) {
        delete[] global_tree_in_array_form_;
        global_tree_in_array_form_ = NULL;
      }
    }

    const TreeType::BoundType &get_node_bound(TreeType * node) const {
      return node->bound();
    }

    TreeType::BoundType &get_node_bound(TreeType * node) {
      return node->bound();
    }

    TreeType *get_node_left_child(TreeType * node) {
      return node->left();
    }

    TreeType *get_node_right_child(TreeType * node) {
      return node->right();
    }

    bool node_is_leaf(TreeType * node) const {
      return node->is_leaf();
    }

    core::tree::AbstractStatistic *&get_node_stat(TreeType * node) {
      return node->stat();
    }

    int get_node_count(TreeType * node) const {
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
      const std::string & file_name,
      boost::mpi::communicator * communicator_in) {

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
      point_inbox_.Init(comm_, &mpi_mutex_);
      point_request_message_box_.Init(comm_, &mpi_mutex_, owned_table_);
    }

    void Save(const std::string & file_name) const {

    }

    void IndexData(
      const core::metric_kernels::AbstractMetric & metric_in,
      double sample_probability_in) {

      // For each process, select a subset of indices to send to the
      // master node.
      printf("Process %d is generating samples to send to Process 0.\n",
             comm_->rank());
      std::vector<int> sampled_indices;
      for(int i = 0; i < owned_table_->n_entries(); i++) {
        if(core::math::Random() <= sample_probability_in) {
          sampled_indices.push_back(i);
        }
      }

      // The number of maximum leaf nodes for the top tree is equal to
      // the number of machines.
      int max_num_leaf_nodes = comm_->size();

      // The maximum number of nodes in the tree given the specified
      // number of maximum leaf nodes.
      int num_nodes;

      std::vector< std::vector<int> > list_of_sampled_indices;
      int total_num_sample_points = 0;

      // For the master node,
      if(comm_->rank() == 0) {

        // Find out the list of points sampled so that we can build a
        // sampled table to build the tree from.
        mpi_mutex_.lock();
        boost::mpi::gather(
          *comm_, sampled_indices, list_of_sampled_indices, 0);
        mpi_mutex_.unlock();

        // Gather all the necessary data from all of the proceses.
        for(unsigned int i = 0; i < list_of_sampled_indices.size(); i++) {
          total_num_sample_points += list_of_sampled_indices[i].size();
        }
      }
      else {

        // Send the list of sampled indices to the master.
        mpi_mutex_.lock();
        boost::mpi::gather(*comm_, sampled_indices, 0);
        mpi_mutex_.unlock();
      }

      core::table::Table sampled_table;
      if(comm_->rank() == 0) {
        sampled_table.Init(this->n_attributes(), total_num_sample_points);
        core::table::DenseMatrix &sampled_table_data = sampled_table.data();

        int column_index = 0;
        for(unsigned int i = 0; i < list_of_sampled_indices.size(); i++) {
          const std::vector<int> &sampled_indices_per_process =
            list_of_sampled_indices[i];
          for(unsigned int j = 0; j < sampled_indices_per_process.size();
              j++, column_index++) {
            this->get(
              i, sampled_indices_per_process[j],
              sampled_table_data.GetColumnPtr(column_index));
          }
        }
        printf("Process 0 collected %d samples across all processes.\n",
               sampled_table.n_entries());
      }

      if(comm_->rank() == 0) {

        core::table::DenseMatrix &sampled_table_data = sampled_table.data();

        // Build the tree.
        std::vector<int> global_old_from_new, global_new_from_old;
        global_tree_ = core::tree::MakeGenMetricTree<TreeType>(
                         metric_in, sampled_table_data, 2,
                         max_num_leaf_nodes,
                         &global_old_from_new, &global_new_from_old,
                         &num_nodes);
        printf("Process 0 finished building the top tree.\n");

        // Broadcast the number of nodes to all processes.
        boost::mpi::broadcast(*comm_, num_nodes, 0);

        // Broadcast the top tree to all the other processes by doing
        // an in-order traversal.
        BroadcastTree_(global_tree_);
      }

      // For the other nodes,
      else {

        // Get the number of nodes from the master node.
        boost::mpi::broadcast(*comm_, num_nodes, 0);

        // Receive back the global tree from the master tree and make
        // a copy.
        global_tree_in_array_form_ = new TreeType[ num_nodes ];
        for(int i = 0; i < num_nodes; i++) {
          mpi_mutex_.lock();
          boost::mpi::broadcast(*comm_, global_tree_in_array_form_[i], 0);
          mpi_mutex_.unlock();
        }
      }

      comm_->barrier();
    }

    void get(
      int requested_rank, int point_id,
      double *entry_out) {

      // If owned by the process, just return the point. Otherwise, we
      // need to send an MPI request to the process holding the
      // required resource.
      if(comm_->rank() == requested_rank) {
        owned_table_->get(point_id, entry_out);
      }
      else {

        // The point request message.
        core::table::PointRequestMessage point_request_message(
          comm_->rank(), point_id);

        // Inform the source processor that this processor needs data!
        {
          boost::unique_lock<boost::mutex> lock_in(mpi_mutex_);
          comm_->isend(
            requested_rank,
            core::table::DistributedTableMessage::REQUEST_POINT,
            point_request_message);
        }

        // Do a conditional wait until the point is ready.
        boost::unique_lock<boost::mutex> lock(
          point_inbox_.point_received_mutex());
        point_inbox_.wait(lock);

        // If we are here, then the point is ready. Copy the point.
        for(int i = 0; i < this->n_attributes(); i++) {
          entry_out[i] = point_inbox_.point()[i];
        }

        // Signal that we are done copying out the point.
        point_inbox_.invalidate_point();
      }
    }

    void get(
      int requested_rank, int point_id,
      core::table::DensePoint * entry) {

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
        {
          boost::unique_lock<boost::mutex> lock_in(mpi_mutex_);
          comm_->isend(
            requested_rank,
            core::table::DistributedTableMessage::REQUEST_POINT,
            point_request_message);
        }

        // Do a conditional wait until the point is ready.
        boost::unique_lock<boost::mutex> lock(
          point_inbox_.point_received_mutex());
        point_inbox_.wait(lock);

        // If we are here, then the point is ready. Copy the point.
        entry->Init(point_inbox_.point());

        // Signal that we are done copying out the point.
        point_inbox_.invalidate_point();
      }
    }

    void PrintTree() const {
      global_tree_->Print();
    }
};
};
};

#endif

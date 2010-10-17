/** @file distributed_table.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_DISTRIBUTED_TABLE_H
#define CORE_TABLE_DISTRIBUTED_TABLE_H

#include <armadillo>
#include <new>
#include "boost/mpi.hpp"
#include "boost/mpi/collectives.hpp"
#include "boost/thread.hpp"
#include "boost/serialization/string.hpp"
#include "core/table/table.h"
#include "core/table/distributed_table_message.h"
#include "core/table/point_request_message.h"
#include "core/table/mailbox.h"
#include "core/table/memory_mapped_file.h"

namespace core {
namespace table {
class DistributedTable: public boost::noncopyable {

  public:
    typedef core::tree::GeneralBinarySpaceTree < core::tree::BallBound <
    core::table::DensePoint > > TreeType;

  private:

    core::table::MemoryMappedFile global_m_file_;

    /** @brief The mutex for each MPI call.
     */
    boost::mutex mpi_mutex_;

    core::table::Table *owned_table_;

    int *local_n_entries_;

    TreeType *global_tree_;

    std::vector< TreeType * > global_tree_leaf_nodes_;

    boost::mpi::communicator *global_comm_;

    boost::mpi::communicator *table_group_comm_;

    core::table::PointInbox point_inbox_;

    core::table::PointRequestMessageBox point_request_message_box_;

  private:

    void GatherLeafNodes_(
      TreeType *node, std::vector<TreeType *> &leaf_nodes) {

      if(node->is_leaf()) {
        leaf_nodes.push_back(node);
      }
      else {
        GatherLeafNodes_(node->left(), leaf_nodes);
        GatherLeafNodes_(node->right(), leaf_nodes);
      }
    }

    void AssignPointsToLeafNode_(
      const core::metric_kernels::AbstractMetric &metric_in) {

      // Wait until every process gets here.
      global_comm_->barrier();

      printf("Process %d is in AssignPointsToLeafNode_.\n", global_comm_->rank());

      // Gather the leaf nodes.
      GatherLeafNodes_(global_tree_, global_tree_leaf_nodes_);

      // The leaf node assignment for each point.
      std::vector<int> leaf_assignments(owned_table_->n_entries(), 0);

      for(int i = 0; i < owned_table_->n_entries(); i++) {
        core::table::DensePoint point;
        owned_table_->get(i, &point);

        double min_squared_distance = std::numeric_limits<double>::max();
        for(unsigned int j = 0; j < global_tree_leaf_nodes_.size(); j++) {
          const TreeType::BoundType &bound =
            global_tree_leaf_nodes_[j]->bound();
          double squared_distance = metric_in.DistanceSq(point, bound.center());
          if(squared_distance < min_squared_distance) {
            min_squared_distance = squared_distance;
            leaf_assignments[i] = j;
          }
        }
      }
      int *count_distribution = new int[ global_comm_->size()];
      memset(count_distribution, 0, sizeof(int) * global_comm_->size());
      for(unsigned int i = 0; i < leaf_assignments.size(); i++) {
        count_distribution[ leaf_assignments[i] ]++;
      }

      // Do an all-reduce.
      mpi_mutex_.lock();
      boost::mpi::all_reduce(
        *global_comm_, count_distribution, global_comm_->size(),
        local_n_entries_, std::plus<int>());
      delete count_distribution;
      mpi_mutex_.unlock();

      printf("Process %d:\n", global_comm_->rank());
      for(int i = 0; i < global_comm_->size(); i++) {
        printf("%d ", local_n_entries_[i]);
      }
      printf("\n");
    }

    void DistributeTree_(
      const core::metric_kernels::AbstractMetric & metric_in,
      int max_num_leaf_nodes,
      core::table::Table &sampled_table) {

      // Wait until every process gets here.
      global_comm_->barrier();

      if(global_comm_->rank() == 0) {

        printf("Process 0 is building the tree and distributing the tree.\n");

        int num_nodes;
        core::table::DenseMatrix &sampled_table_data = sampled_table.data();

        // Build the tree.
        std::vector<int> global_old_from_new, global_new_from_old;
        global_tree_ = core::tree::MakeGenMetricTree<TreeType>(
                         metric_in, sampled_table_data, 2,
                         max_num_leaf_nodes,
                         &global_old_from_new, &global_new_from_old,
                         &num_nodes, &global_m_file_);
        printf("Process 0 finished building the top tree with %d nodes.\n",
               num_nodes);

        // Broadcast the top tree to all the other processes by doing
        // an in-order traversal.
        mpi_mutex_.lock();
        boost::mpi::broadcast(* global_comm_, *global_tree_, 0);
        mpi_mutex_.unlock();
      }

      // For the other nodes,
      else {

        printf("Process %d is receiving the tree.\n", global_comm_->rank());

        // Receive back the global tree from the master tree and make
        // a copy.
        mpi_mutex_.lock();
        global_tree_ = new TreeType();
        boost::mpi::broadcast(*global_comm_, *global_tree_, 0);
        mpi_mutex_.unlock();
      }
    }

  public:

    int rank() const {
      return global_comm_->rank();
    }

    bool IsIndexed() const {
      return global_tree_ != NULL;
    }

    DistributedTable() {
      global_comm_ = NULL;
      owned_table_ = NULL;
      global_tree_ = NULL;
    }

    ~DistributedTable() {

      /*
      // Terminate the point request message box.
      {
        {
          boost::unique_lock<boost::mutex> lock_in(mpi_mutex_);
          global_comm_->isend(
            global_comm_->rank(),
            core::table::DistributedTableMessage::TERMINATE_POINT_REQUEST_MESSAGE_BOX, 0);
        }
        point_request_message_box_.Join();
      }

      // Wait until the point inbox is destroyed.
      {
        {
          boost::unique_lock<boost::mutex> lock_in(mpi_mutex_);
          global_comm_->isend(
            global_comm_->rank(),
            core::table::DistributedTableMessage::TERMINATE_POINT_INBOX, 0);
        }
        point_inbox_.Join();
      }
      */

      // Put a barrier so that all processes owning a part of a
      // distributed table are ready to destroy.
      table_group_comm_->barrier();

      // Delete the list of number of entries for each table in the
      // distributed table.
      if(local_n_entries_ != NULL) {
        delete local_n_entries_;
        local_n_entries_ = NULL;
      }

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
      boost::mpi::communicator * global_communicator_in,
      boost::mpi::communicator *table_group_communicator_in) {

      // Set the communicators and read the table.
      global_comm_ = global_communicator_in;
      table_group_comm_ = table_group_communicator_in;
      owned_table_ = new core::table::Table();
      owned_table_->Init(file_name, &global_m_file_);

      // Allocate the vector for storing the number of entries for all
      // the tables in the world, and do an all-gather operation to
      // find out all the sizes.
      local_n_entries_ = new int[ table_group_comm_->size()];
      boost::mpi::all_gather(
        *table_group_comm_, owned_table_->n_entries(), local_n_entries_);

      // Initialize the mail boxes.
      // point_inbox_.Init(global_comm_, &mpi_mutex_);
      // point_request_message_box_.Init(global_comm_, &mpi_mutex_, owned_table_);
    }

    void Save(const std::string & file_name) const {

    }

    void IndexData(
      const core::metric_kernels::AbstractMetric & metric_in,
      double sample_probability_in) {

      // For each process, select a subset of indices to send to the
      // master node.
      printf("Process %d is generating samples to send to Process 0.\n",
             global_comm_->rank());
      std::vector<int> sampled_indices;
      for(int i = 0; i < owned_table_->n_entries(); i++) {
        if(core::math::Random() <= sample_probability_in) {
          sampled_indices.push_back(i);
        }
      }

      // The number of maximum leaf nodes for the top tree is equal to
      // the number of machines.
      int max_num_leaf_nodes = global_comm_->size();

      std::vector< std::vector<int> > list_of_sampled_indices;
      int total_num_sample_points = 0;

      // For the master node,
      if(global_comm_->rank() == 0) {

        // Find out the list of points sampled so that we can build a
        // sampled table to build the tree from.
        mpi_mutex_.lock();
        boost::mpi::gather(
          *global_comm_, sampled_indices, list_of_sampled_indices, 0);
        mpi_mutex_.unlock();

        // Gather all the necessary data from all of the proceses.
        for(unsigned int i = 0; i < list_of_sampled_indices.size(); i++) {
          total_num_sample_points += list_of_sampled_indices[i].size();
        }
      }
      else {

        // Send the list of sampled indices to the master.
        mpi_mutex_.lock();
        boost::mpi::gather(*global_comm_, sampled_indices, 0);
        mpi_mutex_.unlock();
      }

      // Wait until every process gets here.
      global_comm_->barrier();

      core::table::Table sampled_table;
      if(global_comm_->rank() == 0) {
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

      // Get the tree.
      DistributeTree_(metric_in, max_num_leaf_nodes, sampled_table);

      // Now we need to let each process figure out which leaf node it
      // wants to have.
      AssignPointsToLeafNode_(metric_in);

      // Put a barrier.
      global_comm_->barrier();

      // Each process reallocates the table.
      core::table::Table *new_table = new core::table::Table();
      new_table->Init(this->n_attributes(), local_n_entries_[ global_comm_->rank()]);

      // Put a barrier.
      global_comm_->barrier();
    }

    void get(
      int requested_rank, int point_id,
      double *entry_out) {

      // If owned by the process, just return the point. Otherwise, we
      // need to send an MPI request to the process holding the
      // required resource.
      if(global_comm_->rank() == requested_rank) {
        owned_table_->get(point_id, entry_out);
      }
      else {

        // The point request message.
        core::table::PointRequestMessage point_request_message(
          global_comm_->rank(), point_id);

        // Inform the source processor that this processor needs data!
        {
          boost::unique_lock<boost::mutex> lock_in(mpi_mutex_);
          global_comm_->isend(
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
      if(global_comm_->rank() == requested_rank) {
        owned_table_->get(point_id, entry);
      }
      else {

        // The point request message.
        core::table::PointRequestMessage point_request_message(
          global_comm_->rank(), point_id);

        // Inform the source processor that this processor needs data!
        {
          boost::unique_lock<boost::mutex> lock_in(mpi_mutex_);
          global_comm_->isend(
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

/** @file distributed_table.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_DISTRIBUTED_TABLE_H
#define CORE_TABLE_DISTRIBUTED_TABLE_H

#include <armadillo>
#include <new>
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/serialization/string.hpp>
#include <boost/interprocess/offset_ptr.hpp>
#include <boost/random/variate_generator.hpp>
#include "core/table/table.h"
#include "core/table/mailbox.h"
#include "core/table/distributed_table_message.h"
#include "core/table/point_request_message.h"
#include "core/table/memory_mapped_file.h"
#include "core/tree/gen_metric_tree.h"
#include "core/table/distributed_auction.h"
#include "core/tree/distributed_local_kmeans.h"

namespace core {
namespace table {

extern MemoryMappedFile *global_m_file_;

template<typename TreeSpecType>
class DistributedTable: public boost::noncopyable {

  public:

    typedef core::tree::GeneralBinarySpaceTree <TreeSpecType> TreeType;

    typedef core::table::Table<TreeSpecType> TableType;

  private:

    boost::interprocess::offset_ptr<core::table::TableInbox> table_inbox_;

    boost::interprocess::offset_ptr <
    core::table::TableOutbox<TableType> > table_outbox_;

    boost::interprocess::offset_ptr<TableType> owned_table_;

    boost::interprocess::offset_ptr<int> local_n_entries_;

    boost::interprocess::offset_ptr<TreeType> global_tree_;

    int table_outbox_group_comm_size_;

  private:

    void ReadjustCentroids_(
      boost::mpi::communicator &table_outbox_group_comm,
      const std::vector<TreeType *> &top_leaf_nodes,
      int leaf_node_assignment_index) {


    }

    int TakeLeafNodeOwnerShip_(
      boost::mpi::communicator &table_outbox_group_comm,
      const std::vector<double> &num_points_assigned_to_leaf_nodes) {

      core::table::DistributedAuction auction;
      return auction.Assign(
               table_outbox_group_comm, num_points_assigned_to_leaf_nodes,
               std::numeric_limits<double>::epsilon());
    }

    void GetLeafNodeMembershipCounts_(
      const core::metric_kernels::AbstractMetric &metric_in,
      const std::vector<TreeType *> &top_leaf_nodes,
      std::vector<double> &points_assigned_to_node) {

      for(unsigned int i = 0; i < top_leaf_nodes.size(); i++) {
        points_assigned_to_node[i] = 0;
      }

      // Loop through each point and find the closest leaf node.
      for(int i = 0; i < owned_table_->n_entries(); i++) {
        core::table::DensePoint point;
        owned_table_->get(i, &point);

        // Loop through each leaf node.
        double min_squared_mid_distance = std::numeric_limits<double>::max();
        int min_index = -1;
        for(unsigned int j = 0; j < top_leaf_nodes.size(); j++) {
          const typename TreeType::BoundType &leaf_node_bound =
            top_leaf_nodes[j]->bound();

          // Compute the squared mid-distance.
          double squared_mid_distance = leaf_node_bound.MidDistanceSq(
                                          metric_in, point);
          if(squared_mid_distance < min_squared_mid_distance) {
            min_squared_mid_distance = squared_mid_distance;
            min_index = j;
          }
        }

        // Output the assignments.
        points_assigned_to_node[min_index] += 1.0;
      }
    }

    void SelectSubset_(
      double sample_probability_in, std::vector<int> *sampled_indices_out) {

      std::vector<int> indices(owned_table_->n_entries(), 0);
      for(unsigned int i = 0; i < indices.size(); i++) {
        indices[i] = i;
      }
      int num_elements =
        std::max(
          (int) floor(sample_probability_in * owned_table_->n_entries()), 1);
      for(int i = 0; i < num_elements; i++) {
        int random_index = core::math::RandInt(i, (int) indices.size());
        std::swap(indices[i], indices[ random_index ]);
      }

      for(int i = 0; i < num_elements; i++) {
        sampled_indices_out->push_back(indices[i]);
      }
    }

    void CopyPointsIntoTemporaryBuffer_(
      const std::vector<int> &sampled_indices, double **tmp_buffer) {

      *tmp_buffer = new double[ sampled_indices.size() * this->n_attributes()];

      for(unsigned int i = 0; i < sampled_indices.size(); i++) {
        owned_table_->get(
          sampled_indices[i], (*tmp_buffer) + i * this->n_attributes());
      }
    }

  public:

    void UnlockPointinTableInbox() {
      table_inbox_->UnlockPoint();
    }

    void RunInbox(
      boost::mpi::intercommunicator &inbox_to_outbox_comm_in,
      boost::mpi::intercommunicator &inbox_to_computation_comm_in) {
      table_inbox_->Run(
        inbox_to_outbox_comm_in, inbox_to_computation_comm_in);
    }

    void RunOutbox(
      boost::mpi::intercommunicator &outbox_to_inbox_comm_in,
      boost::mpi::intercommunicator &outbox_to_computation_comm_in) {
      table_outbox_->Run(
        outbox_to_inbox_comm_in, outbox_to_computation_comm_in);
    }

    bool IsIndexed() const {
      return global_tree_ != NULL;
    }

    DistributedTable() {
      table_inbox_ = NULL;
      table_outbox_ = NULL;
      owned_table_ = NULL;
      local_n_entries_ = NULL;
      global_tree_ = NULL;
      table_outbox_group_comm_size_ = -1;
    }

    ~DistributedTable() {

      // Delete the mailboxes.
      if(table_outbox_ != NULL) {
        core::table::global_m_file_->DestroyPtr(table_outbox_.get());
        table_outbox_ = NULL;
      }
      if(table_inbox_ != NULL) {
        core::table::global_m_file_->DestroyPtr(table_inbox_.get());
        table_inbox_ = NULL;
      }

      // Delete the list of number of entries for each table in the
      // distributed table.
      if(local_n_entries_ != NULL) {
        if(core::table::global_m_file_) {
          core::table::global_m_file_->DestroyPtr(local_n_entries_.get());
        }
        else {
          delete[] local_n_entries_.get();
        }
        local_n_entries_ = NULL;
      }

      // Delete the table.
      if(owned_table_ != NULL) {
        if(core::table::global_m_file_) {
          core::table::global_m_file_->DestroyPtr(owned_table_.get());
        }
        else {
          delete owned_table_.get();
        }
        owned_table_ = NULL;
      }

      // Delete the tree.
      if(global_tree_ != NULL) {
        delete global_tree_.get();
        global_tree_ = NULL;
      }
    }

    const typename TreeType::BoundType &get_node_bound(TreeType * node) const {
      return node->bound();
    }

    typename TreeType::BoundType &get_node_bound(TreeType * node) {
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

    typename TreeSpecType::StatisticType &get_node_stat(TreeType * node) {
      return node->stat();
    }

    int get_node_count(TreeType * node) const {
      return node->count();
    }

    TreeType *get_tree() {
      return global_tree_.get();
    }

    int n_attributes() const {
      return owned_table_->n_attributes();
    }

    int local_n_entries(int rank_in) const {
      if(rank_in >= table_outbox_group_comm_size_) {
        printf(
          "Invalid rank specified: %d. %d is the limit.\n",
          rank_in, table_outbox_group_comm_size_);
        return -1;
      }
      return local_n_entries_[rank_in];
    }

    int local_n_entries() const {
      return owned_table_->n_entries();
    }

    void Init(
      const std::string & file_name,
      boost::mpi::communicator &table_outbox_group_communicator_in) {

      // Initialize the table owned by the distributed table.
      owned_table_ = (core::table::global_m_file_) ?
                     core::table::global_m_file_->Construct<TableType>() :
                     new TableType();
      owned_table_->Init(file_name);

      // Initialize the mailboxes.
      table_outbox_ = core::table::global_m_file_->Construct <
                      core::table::TableOutbox<TableType> > ();
      table_outbox_->Init(owned_table_);
      table_inbox_ = core::table::global_m_file_->Construct <
                     core::table::TableInbox > ();
      table_inbox_->Init(owned_table_->n_attributes());

      // Allocate the vector for storing the number of entries for all
      // the tables in the world, and do an all-gather operation to
      // find out all the sizes.
      table_outbox_group_comm_size_ = table_outbox_group_communicator_in.size();
      local_n_entries_ = (core::table::global_m_file_) ?
                         (int *) global_m_file_->ConstructArray<int>(
                           table_outbox_group_communicator_in.size()) :
                         new int[ table_outbox_group_communicator_in.size()];
      boost::mpi::all_gather(
        table_outbox_group_communicator_in, owned_table_->n_entries(),
        local_n_entries_.get());
    }

    void Save(const std::string & file_name) const {

    }

    void IndexData(
      const core::metric_kernels::AbstractMetric & metric_in,
      boost::mpi::communicator &table_outbox_group_comm,
      int leaf_size, double sample_probability_in) {

      // Each process generates a random subset of the data points to
      // send to the master. This is a MPI gather operation.
      TableType sampled_table;
      std::vector<int> sampled_indices;
      SelectSubset_(sample_probability_in, &sampled_indices);
      printf("Process %d sampled %d points.\n",
             table_outbox_group_comm.rank(), sampled_indices.size());

      // Send the number of points chosen in this process to the
      // master so that the master can allocate the appropriate amount
      // of space to receive all the points.
      int total_num_samples = 0;
      double *tmp_buffer = NULL;
      int *counts = new int[ table_outbox_group_comm.size()];
      int local_sampled_indices_size = static_cast<int>(
                                         sampled_indices.size());
      boost::mpi::gather(
        table_outbox_group_comm, local_sampled_indices_size, counts, 0);
      for(int i = 0; i < table_outbox_group_comm.size(); i++) {
        total_num_samples += counts[i];
      }

      // Each process copies the subset of points into the temporary
      // buffer.
      CopyPointsIntoTemporaryBuffer_(sampled_indices, &tmp_buffer);

      if(table_outbox_group_comm.rank() == 0) {
        sampled_table.Init(this->n_attributes(), total_num_samples);
        memcpy(
          sampled_table.data().ptr(), tmp_buffer,
          sizeof(double) * this->n_attributes() * sampled_indices.size());

        int offset = this->n_attributes() * sampled_indices.size();
        for(int i = 1; i < table_outbox_group_comm.size(); i++) {
          int num_elements_to_receive = this->n_attributes() * counts[i];
          table_outbox_group_comm.recv(
            i, boost::mpi::any_tag,
            sampled_table.data().ptr() + offset, num_elements_to_receive);
          offset += num_elements_to_receive;
        }
      }
      else {
        table_outbox_group_comm.send(
          0, table_outbox_group_comm.rank(), tmp_buffer,
          this->n_attributes() * sampled_indices.size());
      }

      table_outbox_group_comm.barrier();

      // After sending, free the temporary buffer.
      delete[] tmp_buffer;
      delete[] counts;

      // The master builds the top tree, and sends the leaf nodes to
      // the rest.
      std::vector<TreeType *> top_leaf_nodes;
      if(table_outbox_group_comm.rank() == 0) {

        printf("Building the tree\n");
        sampled_table.data().Print();
        sampled_table.IndexData(
          metric_in, leaf_size, table_outbox_group_comm.size());

        // Broadcast the leaf nodes.
        sampled_table.get_leaf_nodes(
          sampled_table.get_tree(), &top_leaf_nodes);
      }

      // Broadcast the leaf nodes.
      boost::mpi::broadcast(table_outbox_group_comm, top_leaf_nodes, 0);

      printf("Checking the nodes: %d %d\n", table_outbox_group_comm.rank(),
             top_leaf_nodes.size());

      for(unsigned int i = 0; i < top_leaf_nodes.size(); i++) {
        printf("%d %d %d\n", top_leaf_nodes[i]->begin(),
               top_leaf_nodes[i]->end(), top_leaf_nodes[i]->count());
      }

      // Assign each point to one of the leaf nodes.
      std::vector<double> num_points_assigned_to_leaf_nodes(
        top_leaf_nodes.size(), 0);
      GetLeafNodeMembershipCounts_(
        metric_in, top_leaf_nodes, num_points_assigned_to_leaf_nodes);

      // Each process takes a node in a greedy fashion to minimize the
      // data movement.
      int leaf_node_assignment_index = TakeLeafNodeOwnerShip_(
                                         table_outbox_group_comm,
                                         num_points_assigned_to_leaf_nodes);
      printf("Process %d got %d\n", table_outbox_group_comm.rank(),
             leaf_node_assignment_index);

      // Each process decides to test against the assigned leaf and
      // its immediate DFS neighbors.
      ReadjustCentroids_(
        table_outbox_group_comm, top_leaf_nodes, leaf_node_assignment_index);
    }

    void get(
      boost::mpi::intercommunicator &computation_to_outbox_comm_in,
      boost::mpi::intercommunicator &computation_to_inbox_comm_in,
      int requested_rank, int point_id,
      core::table::DensePoint * entry) {

      // If owned by the process, just return the point. Otherwise, we
      // need to send an MPI request to the process holding the
      // required resource.
      if(computation_to_outbox_comm_in.local_rank() == requested_rank) {
        owned_table_->get(point_id, entry);
      }

      // If the inbox has already fetched the point (do a cache
      // lookup) here, then no MPI call is necessary.
      else if(false) {

      }

      else {

        // The point request message.
        core::table::PointRequestMessage point_request_message(
          computation_to_outbox_comm_in.local_rank(), point_id);

        // Inform the source processor that this processor needs data!
        computation_to_outbox_comm_in.send(
          requested_rank,
          core::table::DistributedTableMessage::REQUEST_POINT_FROM_TABLE_OUTBOX,
          point_request_message);

        // Wait until the point has arrived.
        int dummy;
        boost::mpi::request recv_request =
          computation_to_inbox_comm_in.irecv(
            computation_to_outbox_comm_in.local_rank(),
            core::table::DistributedTableMessage::
            RECEIVE_POINT_FROM_TABLE_INBOX, dummy);
        recv_request.wait();

        // If we are here, then the point is ready. Alias the point.
        entry->Alias(
          table_inbox_->get_point(requested_rank, point_id),
          owned_table_->n_attributes());
      }
    }

    void PrintTree() const {
      global_tree_->Print();
    }
};
};
};

#endif

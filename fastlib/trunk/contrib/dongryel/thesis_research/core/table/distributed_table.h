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
#include <boost/random/variate_generator.hpp>
#include "core/table/table.h"
#include "core/table/distributed_table_message.h"
#include "core/table/point_request_message.h"
#include "core/table/memory_mapped_file.h"
#include "core/tree/gen_metric_tree.h"

namespace core {
namespace table {
class DistributedTable: public boost::noncopyable {

  public:
    typedef core::tree::GeneralBinarySpaceTree < core::tree::GenMetricTree <
    core::table::DensePoint > > TreeType;

    typedef core::table::Table<TreeType> TableType;

  private:

    TableType *owned_table_;

    int *local_n_entries_;

    TreeType *global_tree_;

    std::vector< TreeType * > global_tree_leaf_nodes_;

    boost::mpi::communicator *global_comm_;

    boost::mpi::communicator *table_outbox_group_comm_;

    boost::mpi::communicator *table_inbox_group_comm_;

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

      // Put a barrier so that all processes owning a part of a
      // distributed table are ready to destroy.
      global_comm_->barrier();

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
      boost::mpi::communicator *global_communicator_in,
      boost::mpi::communicator *table_outbox_group_communicator_in,
      boost::mpi::communicator *table_inbox_group_communicator_in) {

      // Set the communicators and read the table.
      global_comm_ = global_communicator_in;
      table_outbox_group_comm_ = table_outbox_group_communicator_in;
      table_inbox_group_comm_ = table_inbox_group_communicator_in;
      owned_table_ = new TableType();
      owned_table_->Init(file_name);

      // Allocate the vector for storing the number of entries for all
      // the tables in the world, and do an all-gather operation to
      // find out all the sizes.
      local_n_entries_ = new int[ table_outbox_group_comm_->size()];
      boost::mpi::all_gather(
        *table_outbox_group_comm_, owned_table_->n_entries(), local_n_entries_);
    }

    void Save(const std::string & file_name) const {

    }

    void IndexData(
      const core::metric_kernels::AbstractMetric & metric_in,
      double sample_probability_in) {


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
          table_outbox_group_comm_->rank(), point_id);

        // Inform the source processor that this processor needs data!
        table_outbox_group_comm_->send(
          requested_rank,
          core::table::DistributedTableMessage::REQUEST_POINT,
          point_request_message);

        // Wait until the point has arrived.


        // If we are here, then the point is ready. Copy the point.


        // Signal that we are done copying out the point.

      }
    }

    void PrintTree() const {
      global_tree_->Print();
    }
};
};
};

#endif

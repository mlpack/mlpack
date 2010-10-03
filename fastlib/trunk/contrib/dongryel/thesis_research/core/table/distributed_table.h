/** @file distributed_table.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_DISTRIBUTED_TABLE_H
#define CORE_TABLE_DISTRIBUTED_TABLE_H

#include <armadillo>
#include "boost/mpi.hpp"
#include "core/table/table.h"

namespace core {
namespace table {
class DistributedTableMessage {
  public:
    enum DistributedTableRequest { REQUEST_POINT, RECEIVE_POINT };
};

class DistributedTable: public boost::noncopyable {

    typedef core::tree::GeneralBinarySpaceTree < core::tree::BallBound <
    core::table::DensePoint > > TreeType;

  private:

    int rank_;

    core::table::Table *owned_table_;

    int global_n_entries_;

    TreeType *global_tree_;

    boost::mpi::communicator *comm_;

  public:

    int rank() const {
      return rank_;
    }

    bool IsIndexed() const {
      return global_tree_ != NULL;
    }

    DistributedTable() {
      rank_ = -1;
      owned_table_ = NULL;
      global_n_entries_ = 0;
      global_tree_ = NULL;
    }

    ~DistributedTable() {
      if(owned_table_ != NULL) {
        delete owned_table_;
        rank_ = -1;
        owned_table_ = NULL;
      }
      if(global_tree_ != NULL) {
        delete global_tree_;
        global_tree_ = NULL;
      }
      global_n_entries_ = 0;
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

    int local_n_entries() const {
      return owned_table_->n_entries();
    }

    int global_n_entries() const {
      return global_n_entries_;
    }

    void Init(
      int rank_in,
      const std::string &file_name,
      boost::mpi::communicator *communicator_in) {

      rank_ = rank_in;
      comm_ = communicator_in;
      owned_table_ = new core::table::Table();
      owned_table_->Init(file_name);
    }

    void Save(const std::string &file_name) const {

    }

    void IndexData(
      const core::metric_kernels::AbstractMetric &metric_in, int leaf_size) {

    }

    void get(
      int requested_rank, int point_id,
      core::table::DensePoint *entry) const {

      // If owned by the process, just return the point. Otherwise, we
      // need to send an MPI request to the process holding the
      // required resource.
      if(rank_ == requested_rank) {
        owned_table_->get(point_id, entry);
      }
      else {

        // We receive the point in the form of std::vector.
        std::vector<double> received_point_vector;

        // Inform the other processor that this processor needs data!
        boost::mpi::request point_request = comm_->isend(
                                              requested_rank,
                                              core::table::DistributedTableMessage::REQUEST_POINT,
                                              rank_);
        boost::mpi::request point_receive_request =
          comm_->irecv(
            requested_rank,
            core::table::DistributedTableMessage::RECEIVE_POINT,
            received_point_vector);
        entry->Init(received_point_vector);
        point_request.wait();
      }
    }

    void PrintTree() const {
      global_tree_->Print();
    }
};
};
};

#endif

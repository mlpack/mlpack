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

    void PrintTree() const {
      global_tree_->Print();
    }
};
};
};

#endif

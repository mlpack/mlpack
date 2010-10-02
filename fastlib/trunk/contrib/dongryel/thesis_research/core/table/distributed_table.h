/** @file distributed_table.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_DISTRIBUTED_TABLE_H
#define CORE_TABLE_DISTRIBUTED_TABLE_H

#include <armadillo>
#include "boost/utility.hpp"
#include "core/csv_parser/dataset_reader.h"
#include "core/tree/gen_metric_tree.h"
#include "core/tree/statistic.h"
#include "core/table/dense_matrix.h"
#include "core/table/dense_point.h"

namespace core {
namespace table {
class DistributedTable: public boost::noncopyable {

    typedef core::tree::GeneralBinarySpaceTree < core::tree::BallBound <
    core::table::DensePoint > > TreeType;

  private:

    int rank_;

    Table *owned_table_;

    int global_n_entries_;

    TreeType *global_tree_;

  public:

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
        rank = -1;
        owned_table_ = NULL;
      }
      delete tree_;
      global_n_entries_ = 0;
      global_tree_ = NULL;
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
      return tree_;
    }

    int n_attributes() const {
      return data_.n_rows;
    }

    int n_entries() const {
      return global_n_entries_;
    }

    void Init(int num_dimensions_in, int num_points_in) {
      data_.set_size(num_dimensions_in, num_points_in);
      data_.fill(0);
    }

    void Init(const std::string &file_name) {
      core::DatasetReader::ParseDataset(
        file_name, &data_);
    }

    void Save(const std::string &file_name) const {

    }

    template<typename MetricType>
    void IndexData(const MetricType &metric_in, int leaf_size) {
      tree_ = core::tree::MakeGenMetricTree<TreeType>(
                metric_in, data_, leaf_size, &old_from_new_, &new_from_old_);
    }

    void get(int point_id, core::table::DenseConstPoint *point_out) const {
      direct_get_(point_id, point_out);
    }

    void get(int point_id, core::table::DensePoint *point_out) {
      direct_get_(point_id, point_out);
    }

    void PrintTree() const {
      tree_->Print();
    }
};
};
};

#endif

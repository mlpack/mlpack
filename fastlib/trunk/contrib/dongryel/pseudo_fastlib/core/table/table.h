#ifndef CORE_TABLE_TABLE_H
#define CORE_TABLE_TABLE_H

#include <armadillo>
#include "core/csv_parser/dataset_reader.h"
#include "core/tree/gen_metric_tree.h"
#include "core/tree/statistic.h"

namespace core {
namespace table {
class Table {

  public:
    typedef core::tree::GeneralBinarySpaceTree < core::tree::BallBound <
    core::metric_kernels::LMetric<2>, arma::vec > > TreeType;

  public:

    class TreeIterator {
      private:
        int begin_;

        int end_;

        int current_index_;

        const core::table::Table *table_;

      public:

        TreeIterator(const core::table::Table &table, const TreeType *node) {
          table_ = &table;
          begin_ = node->begin();
          end_ = node->end();
          current_index_ = begin_ - 1;
        }

        TreeIterator(const core::table::Table &table, int begin, int count) {
          table_ = &table;
          begin_ = begin;
          end_ = begin + count;
          current_index_ = begin_ - 1;
        }

        bool HasNext() const {
          return current_index_ < end_ - 1;
        }

        void Next(arma::vec *entry, int *point_id) {
          current_index_++;
          table_->direct_get_(current_index_, entry);
          *point_id = table_->direct_get_id_(current_index_);
        }

        void get(int i, arma::vec *entry) {
          table_->direct_get_(begin_ + i, entry);
        }

        void get_id(int i, int *point_id) {
          *point_id = table_->direct_get_id_(begin_ + i);
        }

        void RandomPick(arma::vec *entry) {
          table_->direct_get_(core::math::Random(begin_, end_), entry);
        }

        void RandomPick(arma::vec *entry, int *point_id) {
          *point_id = core::math::Random(begin_, end_);
          table_->direct_get_(*point_id, entry);
        }

        void Reset() {
          current_index_ = begin_ - 1;
        }

        int count() const {
          return end_ - begin_;
        }

        int begin() const {
          return begin_;
        }

        int end() const {
          return end_;
        }
    };

  private:
    arma::mat data_;

    std::vector<int> old_from_new_;

    std::vector<int> new_from_old_;

    TreeType *tree_;

  public:

    Table() {
      tree_ = NULL;
    }

    ~Table() {
      delete tree_;
      tree_ = NULL;
    }

    TreeIterator get_node_iterator(TreeType *node) const {
      return TreeIterator(*this, node);
    }

    TreeIterator get_node_iterator(int begin, int count) const {
      return TreeIterator(*this, begin, count);
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
      return data_.n_cols;
    }

    void Init(const std::string &file_name) {
      core::DatasetReader::ParseDataset(
        file_name, &data_);
    }

    void IndexData(int leaf_size) {
      tree_ = core::tree::MakeGenMetricTree<TreeType>(
                data_, leaf_size, &old_from_new_, &new_from_old_);
    }

    void get(int point_id, arma::vec *point_out) {
      direct_get_(point_id, point_out);
    }

    void PrintTree() const {
      tree_->Print();
    }

  private:

    void direct_get_(int point_id, arma::vec *entry) const {
      if (tree_ == NULL) {
        *entry = data_.col(point_id);
      }
      else {
        *entry = data_.col(new_from_old_[point_id]);
      }
    }

    int direct_get_id_(int point_id) const {
      if (tree_ == NULL) {
        return point_id;
      }
      else {
        return new_from_old_[point_id];
      }
    }
};
};
};

#endif

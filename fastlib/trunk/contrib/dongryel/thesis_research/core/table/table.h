/** @file table.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_TABLE_H
#define CORE_TABLE_TABLE_H

#include <armadillo>
#include "boost/utility.hpp"
#include "core/csv_parser/dataset_reader.h"
#include "core/metric_kernels/abstract_metric.h"
#include "core/tree/gen_metric_tree.h"
#include "core/tree/statistic.h"
#include "core/table/dense_matrix.h"
#include "core/table/dense_point.h"

namespace core {
namespace table {
template<typename TreeSpecType>
class Table: public boost::noncopyable {

  public:
    typedef core::tree::GeneralBinarySpaceTree < TreeSpecType > TreeType;

    typedef core::table::Table<TreeSpecType> TableType;

  public:

    class TreeIterator {
      private:
        int begin_;

        int end_;

        int current_index_;

        const TableType *table_;

      public:

        TreeIterator() {
          begin_ = -1;
          end_ = -1;
          current_index_ = -1;
          table_ = NULL;
        }

        TreeIterator(const TreeIterator &it_in) {
          begin_ = it_in.begin();
          end_ = it_in.end();
          current_index_ = it_in.current_index();
          table_ = it_in.table();
        }

        TreeIterator(const TableType &table, const TreeType *node) {
          table_ = &table;
          begin_ = node->begin();
          end_ = node->end();
          current_index_ = begin_ - 1;
        }

        TreeIterator(const TableType &table, int begin, int count) {
          table_ = &table;
          begin_ = begin;
          end_ = begin + count;
          current_index_ = begin_ - 1;
        }

        const TableType *table() const {
          return table_;
        }

        bool HasNext() const {
          return current_index_ < end_ - 1;
        }

        void Next() {
          current_index_++;
        }

        void Next(core::table::DenseConstPoint *entry, int *point_id) {
          current_index_++;
          table_->iterator_get_(current_index_, entry);
          *point_id = table_->iterator_get_id_(current_index_);
        }

        void get(int i, core::table::DenseConstPoint *entry) {
          table_->iterator_get_(begin_ + i, entry);
        }

        void get_id(int i, int *point_id) {
          *point_id = table_->iterator_get_id_(begin_ + i);
        }

        void RandomPick(core::table::DenseConstPoint *entry) {
          table_->iterator_get_(core::math::Random(begin_, end_), entry);
        }

        void RandomPick(core::table::DenseConstPoint *entry, int *point_id) {
          *point_id = core::math::Random(begin_, end_);
          table_->iterator_get_(*point_id, entry);
        }

        void Reset() {
          current_index_ = begin_ - 1;
        }

        int current_index() const {
          return current_index_;
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
    core::table::DenseMatrix data_;

    std::vector<int> old_from_new_;

    std::vector<int> new_from_old_;

    TreeType *tree_;

  public:

    core::table::DenseMatrix &data() {
      return data_;
    }

    bool IsIndexed() const {
      return tree_ != NULL;
    }

    Table() {
      tree_ = NULL;
    }

    ~Table() {
      if(tree_) {
        if(core::table::global_m_file_) {
          RecursiveDeallocate_(tree_);
        }
        else {
          delete tree_;
        }
      }
      tree_ = NULL;
    }

    TreeIterator get_node_iterator(TreeType *node) {
      return TreeIterator(*this, node);
    }

    TreeIterator get_node_iterator(int begin, int count) {
      return TreeIterator(*this, begin, count);
    }

    const typename TreeType::BoundType &get_node_bound(TreeType *node) const {
      return node->bound();
    }

    typename TreeType::BoundType &get_node_bound(TreeType *node) {
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
      return data_.n_rows();
    }

    int n_entries() const {
      return data_.n_cols();
    }

    void Init(
      int num_dimensions_in, int num_points_in) {
      data_.Init(num_dimensions_in, num_points_in);
    }

    void Init(const std::string &file_name) {
      core::DatasetReader::ParseDataset(file_name, &data_);
    }

    void Save(const std::string &file_name) const {
      FILE *foutput = fopen(file_name.c_str(), "w+");
      for(int j = 0; j < data_.n_cols(); j++) {
        core::table::DenseConstPoint point;
        this->direct_get_(j, &point);
        for(int i = 0; i < data_.n_rows(); i++) {
          fprintf(foutput, "%g", point[i]);
          if(i < data_.n_rows() - 1) {
            fprintf(foutput, ",");
          }
        }
        fprintf(foutput, "\n");
      }
      fclose(foutput);
    }

    void IndexData(
      const core::metric_kernels::AbstractMetric &metric_in, int leaf_size) {
      int num_nodes;
      tree_ = TreeType::MakeTree(
                metric_in, data_, leaf_size, std::numeric_limits<int>::max(),
                &old_from_new_, &new_from_old_, &num_nodes);
    }

    void get(int point_id, double *point_out) const {
      direct_get_(point_id, point_out);
    }

    void get(int point_id, std::vector<double> *point_out) const {
      direct_get_(point_id, point_out);
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

  private:

    void RecursiveDeallocate_(TreeType *node) {
      if(node->is_leaf() == false) {
        RecursiveDeallocate_(node->left());
        RecursiveDeallocate_(node->right());
      }
      core::table::global_m_file_->Deallocate(node);
    }

    void direct_get_(int point_id, double *entry) const {
      if(this->IsIndexed() == false) {
        data_.CopyColumnVector(point_id, entry);
      }
      else {
        data_.CopyColumnVector(new_from_old_[point_id], entry);
      }
    }

    void direct_get_(int point_id, std::vector<double> *entry) const {
      if(this->IsIndexed() == false) {
        data_.MakeColumnVector(point_id, entry);
      }
      else {
        data_.MakeColumnVector(new_from_old_[point_id], entry);
      }
    }

    void direct_get_(
      int point_id, core::table::DenseConstPoint *entry) const {
      if(this->IsIndexed() == false) {
        data_.MakeColumnVector(point_id, entry);
      }
      else {
        data_.MakeColumnVector(new_from_old_[point_id], entry);
      }
    }

    void direct_get_(int point_id, core::table::DensePoint *entry) {
      if(this->IsIndexed() == false) {
        data_.MakeColumnVector(point_id, entry);
      }
      else {
        data_.MakeColumnVector(new_from_old_[point_id], entry);
      }
    }

    void iterator_get_(
      int reordered_position, core::table::DenseConstPoint *entry) const {
      data_.MakeColumnVector(reordered_position, entry);
    }

    void iterator_get_(
      int reordered_position, core::table::DensePoint *entry) {
      data_.MakeColumnVector(reordered_position, entry);
    }

    int iterator_get_id_(int reordered_position) const {
      if(this->IsIndexed() == false) {
        return reordered_position;
      }
      else {
        return old_from_new_[reordered_position];
      }
    }
};
};
};

#endif

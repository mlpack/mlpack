/** @file table.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_TABLE_H
#define CORE_TABLE_TABLE_H

#include <boost/serialization/serialization.hpp>
#include <boost/interprocess/offset_ptr.hpp>
#include <boost/utility.hpp>
#include "core/csv_parser/dataset_reader.h"
#include "core/metric_kernels/abstract_metric.h"
#include "core/tree/general_spacetree.h"
#include "core/tree/gen_metric_tree.h"
#include "core/tree/statistic.h"
#include "core/table/dense_matrix.h"
#include "core/table/dense_point.h"

namespace core {
namespace table {

extern MemoryMappedFile *global_m_file_;

template<typename IndexType>
class IndexUtil {
  public:
    static int Extract(IndexType *array, int position);

    template<typename Archive>
    static void Serialize(Archive &ar, IndexType *array, int position);
};

template<>
class IndexUtil< int > {
  public:
    static int Extract(int *array, int position) {
      return array[position];
    }

    template<typename Archive>
    static void Serialize(Archive &ar, int *array, int position) {
      ar & array[position];
    }
};

template<>
class IndexUtil< std::pair<int, int> > {
  public:
    static int Extract(std::pair<int, int> *array, int position) {
      return array[position].second;
    }

    template<typename Archive>
    static void Serialize(
      Archive &ar, std::pair<int, int> *array, int position) {
      ar & array[position].first;
      ar & array[position].second;
    }
};

template < typename TreeSpecType, typename IndexType = int >
class Table: public boost::noncopyable {

  public:
    typedef core::tree::GeneralBinarySpaceTree < TreeSpecType > TreeType;

    typedef core::table::Table<TreeSpecType> TableType;

    typedef typename TreeSpecType::StatisticType StatisticType;

  private:
    friend class boost::serialization::access;

    core::table::DenseMatrix data_;

    int rank_;

    boost::interprocess::offset_ptr<IndexType> old_from_new_;

    boost::interprocess::offset_ptr<IndexType> new_from_old_;

    boost::interprocess::offset_ptr<TreeType> tree_;

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

        void Next(core::table::DensePoint *entry, int *point_id) {
          current_index_++;
          table_->iterator_get_(current_index_, entry);
          *point_id = table_->iterator_get_id_(current_index_);
        }

        void get(int i, core::table::DensePoint *entry) {
          table_->iterator_get_(begin_ + i, entry);
        }

        void get_id(int i, int *point_id) {
          *point_id = table_->iterator_get_id_(begin_ + i);
        }

        void RandomPick(core::table::DensePoint *entry) {
          table_->iterator_get_(core::math::Random(begin_, end_), entry);
        }

        void RandomPick(core::table::DensePoint *entry, int *point_id) {
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

  public:

    template<class Archive>
    void save(Archive &ar, const unsigned int version) const {

      // Save the matrix and the rank.
      ar & data_;
      ar & rank_;

      // Save the old_from_new_mapping manually.
      for(int i = 0; i < data_.n_cols(); i++) {
        core::table::IndexUtil<IndexType>::Serialize(
          ar, old_from_new_.get(), i);
      }
      for(int i = 0; i < data_.n_cols(); i++) {
        core::table::IndexUtil<IndexType>::Serialize(
          ar, new_from_old_.get(), i);
      }

      // Save the tree.
      int num_nodes;
      int tree_depth = FindTreeDepth_(tree_.get());
      std::vector< TreeType *> tree_nodes((1 << tree_depth) - 1,  NULL);
      FillTreeNodes_(tree_.get(), 0, tree_nodes, &num_nodes);
      ar & tree_nodes.size();
      ar & num_nodes;
      for(unsigned int i = 0; i < tree_nodes.size(); i++) {
        if(tree_nodes[i]) {
          ar & i & (*(tree_nodes[i]));
        }
      }
    }

    template<class Archive>
    void load(Archive &ar, const unsigned int version) {

      // Load the matrix and the rank.
      ar & data_;
      ar & rank_;

      // Load the mappings manually.
      old_from_new_ = (core::table::global_m_file_) ?
                      core::table::global_m_file_->ConstructArray <
                      IndexType > (data_.n_cols()) :
                      new IndexType[ data_.n_cols()];
      new_from_old_ = (core::table::global_m_file_) ?
                      core::table::global_m_file_->ConstructArray <
                      IndexType > (data_.n_cols()) :
                      new IndexType[ data_.n_cols()];
      for(int i = 0; i < data_.n_cols(); i++) {
        core::table::IndexUtil<IndexType>::Serialize(
          ar, old_from_new_.get(), i);
      }
      for(int i = 0; i < data_.n_cols(); i++) {
        core::table::IndexUtil<IndexType>::Serialize(
          ar, new_from_old_.get(), i);
      }

      // Load up the max number of loads to receive.
      int max_num_nodes;
      int num_nodes;
      ar & max_num_nodes;
      ar & num_nodes;
      std::vector< TreeType *> tree_nodes(max_num_nodes, NULL);
      for(int i = 0; i < num_nodes; i++) {
        int node_index;
        ar & node_index;
        tree_nodes[node_index] =
          (core::table::global_m_file_) ?
          core::table::global_m_file_->Construct<TreeType>() : new TreeType();
        ar & (*tree_nodes[node_index]);
      }

      // Do the pointer corrections, and have the tree point to the
      // 0-th element.
      for(unsigned int i = 0; i < tree_nodes.size(); i++) {
        if(tree_nodes[i]) {
          tree_nodes[i]->set_children(
            data_, tree_nodes[2 * i + 1], tree_nodes[2 * i + 2]);
        }
      }
      tree_ = tree_nodes[0];
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    IndexType *old_from_new() {
      return old_from_new_.get();
    }

    core::table::DenseMatrix &data() {
      return data_;
    }

    bool IsIndexed() const {
      return tree_ != NULL;
    }

    Table() {
      rank_ = 0;
      tree_ = NULL;
      old_from_new_ = NULL;
      new_from_old_ = NULL;
    }

    ~Table() {
      if(tree_.get() != NULL) {
        if(core::table::global_m_file_) {
          core::table::global_m_file_->DestroyPtr(tree_.get());
        }
        else {
          delete tree_.get();
        }
        tree_ = NULL;
      }
      if(old_from_new_.get() != NULL) {
        if(core::table::global_m_file_) {
          core::table::global_m_file_->DestroyPtr(old_from_new_.get());
        }
        else {
          delete[] old_from_new_.get();
        }
        old_from_new_ = NULL;
      }
      if(new_from_old_.get() != NULL) {
        if(core::table::global_m_file_) {
          core::table::global_m_file_->DestroyPtr(new_from_old_.get());
        }
        else {
          delete[] new_from_old_.get();
        }
        new_from_old_ = NULL;
      }
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

    StatisticType &get_node_stat(TreeType *node) {
      return node->stat();
    }

    int get_node_count(TreeType *node) const {
      return node->count();
    }

    TreeType *get_tree() {
      return tree_.get();
    }

    void get_leaf_nodes(
      TreeType *node, std::vector< TreeType *> *leaf_nodes) {
      if(node->is_leaf()) {
        leaf_nodes->push_back(node);
      }
      else {
        get_leaf_nodes(node->left(), leaf_nodes);
        get_leaf_nodes(node->right(), leaf_nodes);
      }
    }

    int n_attributes() const {
      return data_.n_rows();
    }

    int n_entries() const {
      return data_.n_cols();
    }

    void Init(
      int num_dimensions_in, int num_points_in, int rank_in = 0) {
      if(num_dimensions_in > 0 && num_points_in > 0) {
        data_.Init(num_dimensions_in, num_points_in);
      }
      rank_ = rank_in;

      if(core::table::global_m_file_) {
        old_from_new_ = core::table::global_m_file_->ConstructArray<IndexType>(
                          data_.n_cols());
      }
      else {
        old_from_new_ = new IndexType[data_.n_cols()];
      }
    }

    void Init(const std::string &file_name, int rank_in = 0) {
      core::DatasetReader::ParseDataset(file_name, &data_);
      rank_ = rank_in;

      if(core::table::global_m_file_) {
        old_from_new_ = core::table::global_m_file_->ConstructArray<IndexType>(
                          data_.n_cols());
      }
      else {
        old_from_new_ = new IndexType[data_.n_cols()];
      }
    }

    void Save(const std::string &file_name) const {
      FILE *foutput = fopen(file_name.c_str(), "w+");
      for(int j = 0; j < data_.n_cols(); j++) {
        core::table::DensePoint point;
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
      const core::metric_kernels::AbstractMetric &metric_in, int leaf_size,
      int max_num_leaf_nodes = std::numeric_limits<int>::max()) {
      int num_nodes;

      if(core::table::global_m_file_) {
        new_from_old_ = core::table::global_m_file_->ConstructArray<IndexType>(
                          data_.n_cols());
      }
      else {
        new_from_old_ = new IndexType[data_.n_cols()];
      }
      tree_ = TreeType::MakeTree(
                metric_in, data_, leaf_size, old_from_new_.get(),
                new_from_old_.get(), max_num_leaf_nodes, &num_nodes, rank_);
    }

    void get(int point_id, double *point_out) const {
      direct_get_(point_id, point_out);
    }

    void get(int point_id, std::vector<double> *point_out) const {
      direct_get_(point_id, point_out);
    }

    void get(int point_id, core::table::DensePoint *point_out) const {
      direct_get_(point_id, point_out);
    }

    void get(int point_id, core::table::DensePoint *point_out) {
      direct_get_(point_id, point_out);
    }

    void PrintTree() const {
      tree_->Print();
    }

    const double *GetColumnPtr(int point_id) const {
      if(this->IsIndexed() == false) {
        return data_.GetColumnPtr(point_id);
      }
      else {
        return data_.GetColumnPtr(
                 IndexUtil<IndexType>::Extract(new_from_old_.get(), point_id));
      }
    }

  private:

    void FillTreeNodes_(
      TreeType *node, int node_index, std::vector<TreeType *> &sorted_nodes,
      int *num_nodes) {

      if(node != NULL) {
        (*num_nodes)++;
      }
      sorted_nodes[node_index] = node;

      if(node->is_leaf() == false) {
        FillTreeNodes_(
          node->left(), 2 * node_index + 1, sorted_nodes);
        FillTreeNodes_(
          node->right(), 2 * node_index + 2, sorted_nodes);
      }
    }

    int FindTreeDepth_(TreeType *node) {
      if(node == NULL) {
        return 0;
      }
      int left_depth = FindTreeDepth_(node->left());
      int right_depth = FindTreeDepth_(node->right());
      return (left_depth > right_depth) ? (left_depth + 1) : (right_depth + 1);
    }

    void direct_get_(int point_id, double *entry) const {
      if(this->IsIndexed() == false) {
        data_.CopyColumnVector(point_id, entry);
      }
      else {
        data_.CopyColumnVector(
          IndexUtil<IndexType>::Extract(
            new_from_old_.get(), point_id), entry);
      }
    }

    void direct_get_(int point_id, std::vector<double> *entry) const {
      if(this->IsIndexed() == false) {
        data_.MakeColumnVector(point_id, entry);
      }
      else {
        data_.MakeColumnVector(
          IndexUtil<IndexType>::Extract(
            new_from_old_.get(), point_id), entry);
      }
    }

    void direct_get_(
      int point_id, core::table::DensePoint *entry) const {
      if(this->IsIndexed() == false) {
        data_.MakeColumnVector(point_id, entry);
      }
      else {
        data_.MakeColumnVector(
          IndexUtil<IndexType>::Extract(
            new_from_old_.get(), point_id), entry);
      }
    }

    void direct_get_(int point_id, core::table::DensePoint *entry) {
      if(this->IsIndexed() == false) {
        data_.MakeColumnVector(point_id, entry);
      }
      else {
        data_.MakeColumnVector(
          IndexUtil<IndexType>::Extract(
            new_from_old_.get(), point_id), entry);
      }
    }

    void iterator_get_(
      int reordered_position, core::table::DensePoint *entry) const {
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
        return IndexUtil<IndexType>::Extract(
                 old_from_new_.get(), reordered_position);
      }
    }
};
};
};

#endif

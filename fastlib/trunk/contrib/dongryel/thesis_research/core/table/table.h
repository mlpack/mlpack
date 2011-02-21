/** @file table.h
 *
 *  An abstract organization of a multidimensional dataset with its
 *  indexing structure.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_TABLE_H
#define CORE_TABLE_TABLE_H

#include <boost/serialization/serialization.hpp>
#include <boost/interprocess/offset_ptr.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/utility.hpp>
#include "core/csv_parser/dataset_reader.h"
#include "core/math/math_lib.h"
#include "core/tree/general_spacetree.h"
#include "core/table/dense_matrix.h"
#include "core/table/dense_point.h"
#include "core/table/index_util.h"
#include "core/table/sub_table.h"

namespace core {
namespace table {

extern MemoryMappedFile *global_m_file_;

template <
typename IncomingTreeSpecType, typename IncomingOldFromNewIndexType = int >
class Table {

  public:

    typedef IncomingTreeSpecType TreeSpecType;

    typedef IncomingOldFromNewIndexType OldFromNewIndexType;

    typedef core::tree::GeneralBinarySpaceTree < TreeSpecType > TreeType;

    typedef core::table::Table <
    TreeSpecType, OldFromNewIndexType > TableType;

    typedef typename TreeSpecType::StatisticType StatisticType;

  private:

    // For Boost serialization.
    friend class boost::serialization::access;

    /** @brief The flag that tells whether entire points are available
     *         under the root node.
     */
    bool entire_points_available_;

    /** @brief The list of begin/count pairs so that the underlying
     *         points are accessed correctly using node iterators.
     */
    std::vector< boost::tuple<int, int, int> > begin_count_pairs_;

    /** @brief The underlying multidimensional data owned by the
     *         table.
     */
    core::table::DenseMatrix data_;

    /** @brief The weights associated with each point.
     */
    core::table::DensePoint weights_;

    /** @brief The rank of the table.
     */
    int rank_;

    /** @brief The old from new mapping.
     */
    boost::interprocess::offset_ptr<OldFromNewIndexType> old_from_new_;

    /** @brief The new from old mapping.
     */
    boost::interprocess::offset_ptr<int> new_from_old_;

    /** @brief The tree.
     */
    boost::interprocess::offset_ptr<TreeType> tree_;

    /** @brief Whether the tree is an alias of another tree.
     */
    bool tree_is_aliased_;

    /** @brief Whether the old_from_new/new_from_old mappings are
     *         aliased.
     */
    bool mappings_are_aliased_;

  public:

    /** @brief The iterator for the node that belongs to this table.
     */
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

        TreeIterator(const TableType &table, TreeType *node) {
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

        template<typename PointType>
        void Next(PointType *entry, int *point_id) {
          current_index_++;
          table_->iterator_get_(current_index_, entry);
          *point_id = table_->iterator_get_id_(current_index_);
        }

        template<typename PointType>
        void Next(PointType *entry) {
          current_index_++;
          table_->iterator_get_(current_index_, entry);
        }

        template<typename PointType>
        void get(int i, PointType *entry) {
          table_->iterator_get_(begin_ + i, entry);
        }

        void get_id(int i, int *point_id) {
          *point_id = table_->iterator_get_id_(begin_ + i);
        }

        template<typename PointType>
        void RandomPick(PointType *entry) {
          table_->iterator_get_(core::math::RandInt(begin_, end_), entry);
        }

        template<typename PointType>
        void RandomPick(PointType *entry, int *point_id) {
          *point_id = core::math::RandInt(begin_, end_);
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

    bool points_available_underneath(TreeType *node) const {
      if(entire_points_available_) {
        return true;
      }
      int index = -1;
      for(unsigned int i = 0; i < begin_count_pairs_.size(); i++) {
        if(begin_count_pairs_[i].get<0>() <= node->begin() &&
            node->end() <=
            begin_count_pairs_[i].get<0>() +
            begin_count_pairs_[i].get<1>()) {
          index = i;
          break;
        }
      }
      printf("%d %d is contained in %d %d at index %d\n",
             node->begin(), node->count(),
             begin_count_pairs_[index].get<0>(),
             begin_count_pairs_[index].get<0>() +
             begin_count_pairs_[index].get<1>(), index);
      return index >= 0;
    }

    void add_begin_count_pairs(int begin, int count) {

      // By default, we assume that not all points are available, once
      // you call this method.
      entire_points_available_ = false;

      if(begin_count_pairs_.size() == 0) {
        begin_count_pairs_.push_back(
          boost::tuple<int, int, int>(begin, count, 0));
      }
      else {
        int cumulative_index =
          begin_count_pairs_.back().get<2>() +
          begin_count_pairs_.back().get<1>();
        if(begin != begin_count_pairs_.back().get<0>() +
            begin_count_pairs_.back().get<1>()) {
          begin_count_pairs_.push_back(
            boost::tuple<int, int, int>(begin, count, cumulative_index));
        }
        else {
          int prev_begin = begin_count_pairs_.back().get<0>();
          int prev_count = begin_count_pairs_.back().get<1>();
          int prev_cumulative_index = begin_count_pairs_.back().get<2>();
          begin_count_pairs_.back() =
            boost::tuple<int, int, int>(
              prev_begin, prev_count + count, prev_cumulative_index);
        }
      }
    }

    const std::vector <
    boost::tuple<int, int, int> > &begin_counts_pairs() const {
      return begin_count_pairs_;
    }

    bool tree_is_aliased() const {
      return tree_is_aliased_;
    }

    bool mappings_are_aliased() const {
      return mappings_are_aliased_;
    }

    void Alias(
      OldFromNewIndexType *old_from_new_in, int *new_from_old_in) {
      old_from_new_ = old_from_new_in;
      new_from_old_ = new_from_old_in;
      mappings_are_aliased_ = true;
    }

    /** @brief Basically does everything necessary to steal the
     *         ownership of the pointers of the incoming table.
     */
    void operator=(const TableType &table_in) {
      begin_count_pairs_ = table_in.begin_count_pairs();
      data_ = const_cast<TableType &>(table_in).data();
      rank_ = table_in.rank();
      old_from_new_ = const_cast<TableType &>(table_in).old_from_new();
      new_from_old_ = const_cast<TableType &>(table_in).new_from_old();
      tree_ = const_cast<TableType &>(table_in).get_tree();

      // We steal the pointers.
      mappings_are_aliased_ = table_in.mappings_are_aliased();
      tree_is_aliased_ = table_in.tree_is_aliased();
      const_cast<TableType &>(table_in).mappings_are_aliased_ = true;
      const_cast<TableType &>(table_in).tree_is_aliased_ = true;
    }

    /** @brief The copy constructor that steals the ownership of the
     *         incoming table.
     */
    Table(const TableType &table_in) {
      this->operator=(table_in);
    }

    /** @brief Serializes the table.
     */
    template<class Archive>
    void save(Archive &ar, const unsigned int version) const {
      core::table::SubTable<TableType> sub_table;
      TableType *this_table = const_cast<TableType *>(this);
      sub_table.Init(
        this_table, this_table->get_tree(),
        std::numeric_limits<int>::max(), true);
      ar & sub_table;
    }

    /** @brief Unserialize the table.
     */
    template<class Archive>
    void load(Archive &ar, const unsigned int version) {
      core::table::SubTable<TableType> sub_table;
      sub_table.Init(
        this, this->get_tree(), std::numeric_limits<int>::max(), true);
      ar & sub_table;
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    /** @brief Returns the rank of the table.
     */
    int rank() const {
      return rank_;
    }

    /** @brief Sets the rank of the table.
     */
    void set_rank(int rank_in) {
      rank_ = rank_in;
    }

    OldFromNewIndexType *old_from_new() {
      return old_from_new_.get();
    }

    boost::interprocess::offset_ptr <
    OldFromNewIndexType > *old_from_new_offset_ptr() {
      return &old_from_new_;
    }

    int *new_from_old() {
      return new_from_old_.get();
    }

    boost::interprocess::offset_ptr<int> *new_from_old_offset_ptr() {
      return &new_from_old_;
    }

    /** @brief Returns a reference to the underlying data.
     */
    const core::table::DenseMatrix &data() const {
      return data_;
    }

    /** @brief Returns a reference to the underlying data.
     */
    core::table::DenseMatrix &data() {
      return data_;
    }

    /** @brief Returns a reference to the weights associated with the
     *         underlying data.
     */
    const core::table::DensePoint &weights() const {
      return weights_;
    }

    /** @brief Returns whether the tree is indexed or not.
     */
    bool IsIndexed() const {
      return tree_ != NULL;
    }

    /** @brief The default constructor.
     */
    Table() {
      entire_points_available_ = true;
      rank_ = 0;
      tree_ = NULL;
      old_from_new_ = NULL;
      new_from_old_ = NULL;
      mappings_are_aliased_ = false;
      tree_is_aliased_ = false;
    }

    /** @brief The destructor.
     */
    ~Table() {
      if(tree_is_aliased_ == false) {
        if(tree_.get() != NULL) {
          if(core::table::global_m_file_) {
            core::table::global_m_file_->DestroyPtr(tree_.get());
          }
          else {
            delete tree_.get();
          }
          tree_ = NULL;
        }
      }
      if(mappings_are_aliased_ == false) {
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
    }

    /** @brief Gets the iterator for the node.
     */
    TreeIterator get_node_iterator(TreeType *node) {
      return TreeIterator(*this, node);
    }

    /** @brief Gets the iterator for the node.
     */
    TreeIterator get_node_iterator(int begin, int count) {
      return TreeIterator(*this, begin, count);
    }

    /** @brief Returns the tree owned by the table.
     */
    TreeType *get_tree() {
      return tree_.get();
    }

    boost::interprocess::offset_ptr<TreeType> *get_tree_offset_ptr() {
      return &tree_;
    }

    /** @brief Returns the list of all nodes owned by the tree.
     */
    void get_nodes(TreeType *node, std::vector<TreeType *> *nodes_out) {
      if(node != NULL) {
        nodes_out->push_back(node);
      }
      if(! node->is_leaf()) {
        get_nodes(node->left(), nodes_out);
        get_nodes(node->right(), nodes_out);
      }
    }

    /** @brief Returns the leaf nodes of the tree owned by the table.
     */
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

    /** @brief Returns the number of attributes of the data owned by
     *         the table.
     */
    int n_attributes() const {
      return data_.n_rows();
    }

    /** @brief Returns the number of points owned by the table.
     */
    int n_entries() const {
      return data_.n_cols();
    }

    /** @brief Initializes an empty table with the specified number of
     *         dimensions and number of points (optionally with the
     *         rank).
     */
    void Init(
      int num_dimensions_in, int num_points_in, int rank_in = 0) {
      rank_ = rank_in;
      if(num_dimensions_in > 0 && num_points_in > 0) {
        data_.Init(num_dimensions_in, num_points_in);
        if(core::table::global_m_file_) {
          old_from_new_ = core::table::global_m_file_->ConstructArray <
                          OldFromNewIndexType > (
                            data_.n_cols());
          new_from_old_ = core::table::global_m_file_->ConstructArray <
                          int > (
                            data_.n_cols());
        }
        else {
          old_from_new_ = new OldFromNewIndexType[data_.n_cols()];
          new_from_old_ = new int[data_.n_cols()];
        }
        core::tree::IndexInitializer<OldFromNewIndexType>::OldFromNew(
          data_, rank_in, old_from_new_.get());
      }
    }

    /** @brief Initializes the table from a file, optionally its rank.
     */
    void Init(const std::string &file_name, int rank_in = 0) {
      if(core::DatasetReader::ParseDataset(file_name, &data_) == false) {
        exit(0);
      }
      rank_ = rank_in;

      if(core::table::global_m_file_) {
        old_from_new_ = core::table::global_m_file_->ConstructArray <
                        OldFromNewIndexType > (
                          data_.n_cols());
        new_from_old_ = core::table::global_m_file_->ConstructArray <
                        int > (
                          data_.n_cols());
      }
      else {
        old_from_new_ = new OldFromNewIndexType[data_.n_cols()];
        new_from_old_ = new int[data_.n_cols()];
      }
      core::tree::IndexInitializer<OldFromNewIndexType>::OldFromNew(
        data_, rank_in, old_from_new_.get());
    }

    /** @brief Saves the table to a text file.
     */
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

    template<typename MetricType>
    void IndexData(
      const MetricType &metric_in, int leaf_size,
      int max_num_leaf_nodes = std::numeric_limits<int>::max()) {
      int num_nodes;
      tree_ = TreeType::MakeTree(
                metric_in, data_, leaf_size, old_from_new_.get(),
                new_from_old_.get(), max_num_leaf_nodes, &num_nodes, rank_);

      // The following part should really be fixed in the future. What
      // really needs to happen here is that the weights themselves
      // need to be re-shuffled across machines as well. We hard-code
      // uniform weights for now.
      weights_.Init(data_.n_cols());
      for(int i = 0; i < data_.n_cols(); i++) {
        weights_[i] = 1.0;
      }
    }

    template<typename PointType>
    void get(int point_id, PointType *point_out) const {
      direct_get_(point_id, point_out);
    }

    template<typename PointType>
    void get(int point_id, PointType *point_out) {
      direct_get_(point_id, point_out);
    }

    /** @brief Prints the tree owned by the table.
     */
    void PrintTree() const {
      tree_->Print();
    }

    const double *GetColumnPtr(int point_id) const {
      if(this->IsIndexed() == false) {
        return data_.GetColumnPtr(point_id);
      }
      else {
        return data_.GetColumnPtr(
                 IndexUtil<int>::Extract(
                   new_from_old_.get(), point_id));
      }
    }

  private:

    template<typename PointType>
    void direct_get_(
      int point_id, PointType *entry) const {
      if(this->IsIndexed() == false) {
        data_.MakeColumnVector(point_id, entry);
      }
      else {
        data_.MakeColumnVector(
          IndexUtil<int>::Extract(
            new_from_old_.get(), point_id), entry);
      }
    }

    template<typename PointType>
    void direct_get_(int point_id, PointType *entry) {
      if(this->IsIndexed() == false) {
        data_.MakeColumnVector(point_id, entry);
      }
      else {
        data_.MakeColumnVector(
          IndexUtil<int>::Extract(
            new_from_old_.get(), point_id), entry);
      }
    }

    template<typename PointType>
    void iterator_get_(
      int reordered_position, PointType *entry) const {
      data_.MakeColumnVector(
        this->locate_reordered_position_(reordered_position), entry);
    }

    template<typename PointType>
    void iterator_get_(
      int reordered_position, PointType *entry) {
      data_.MakeColumnVector(
        this->locate_reordered_position_(reordered_position), entry);
    }

    int iterator_get_id_(int reordered_position) const {
      if(this->IsIndexed() == false) {
        return reordered_position;
      }
      else {
        return IndexUtil<OldFromNewIndexType>::Extract(
                 old_from_new_.get(),
                 this->locate_reordered_position_(reordered_position));
      }
    }

    int locate_reordered_position_(int reordered_position) const {
      if(begin_count_pairs_.size() == 0 ||
          (begin_count_pairs_.size() == 1 &&
           begin_count_pairs_[0].get<0>() == 0)) {
        return reordered_position;
      }
      int index = -1;
      for(unsigned int i = 0; i < begin_count_pairs_.size(); i++) {
        if(begin_count_pairs_[i].get<0>() <= reordered_position &&
            reordered_position <
            begin_count_pairs_[i].get<0>() +
            begin_count_pairs_[i].get<1>()) {
          index = i;
          break;
        }
      }
      return begin_count_pairs_[index].get<2>() +
             (reordered_position - begin_count_pairs_[index].get<0>());
    }
};
}
}

#endif

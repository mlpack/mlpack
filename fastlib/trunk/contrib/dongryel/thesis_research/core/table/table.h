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
#include <boost/shared_ptr.hpp>
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
typename IncomingTreeSpecType,
         typename IncomingQueryResultType,
         typename IncomingOldFromNewIndexType = int >
class Table {

  public:

    typedef IncomingTreeSpecType TreeSpecType;

    typedef IncomingOldFromNewIndexType OldFromNewIndexType;

    typedef IncomingQueryResultType QueryResultType;

    typedef core::tree::GeneralBinarySpaceTree < TreeSpecType > TreeType;

    typedef core::table::Table <
    TreeSpecType, IncomingQueryResultType, OldFromNewIndexType > TableType;

    typedef typename TreeSpecType::StatisticType StatisticType;

    typedef core::table::SubTable<TableType> SubTableType;

  private:

    // For Boost serialization.
    friend class boost::serialization::access;

    /** @brief The underlying multidimensional data owned by the
     *         table.
     */
    arma::mat data_;

    /** @brief The weights associated with each point.
     */
    arma::mat weights_;

    /** @brief The rank of the table.
     */
    int rank_;

    /** @brief The old from new mapping.
     */
    boost::interprocess::offset_ptr<OldFromNewIndexType> old_from_new_;

    /** @brief The new from old mapping.
     */
    boost::interprocess::offset_ptr<int> new_from_old_;

    std::map<int, int> new_to_position_map_;

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

        void operator=(const TreeIterator &it_in) {
          begin_ = it_in.begin();
          end_ = it_in.end();
          current_index_ = it_in.current_index();
          table_ = it_in.table();
        }

        TreeIterator(const TreeIterator &it_in) {
          this->operator=(it_in);
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

        void Next(
          arma::vec *entry, int *point_id, double *weight) {
          current_index_++;
          table_->iterator_get_(current_index_, entry, weight);
          *point_id = table_->iterator_get_id_(current_index_);
        }

        void Next(arma::vec *entry, double *weight) {
          current_index_++;
          table_->iterator_get_(current_index_, entry, weight);
        }

        void Next(arma::vec *entry, int *point_id) {
          current_index_++;
          table_->iterator_get_(current_index_, entry);
          *point_id = table_->iterator_get_id_(current_index_);
        }

        void Next(arma::vec *entry) {
          current_index_++;
          table_->iterator_get_(current_index_, entry);
        }

        void Next(int *point_id) {
          current_index_++;
          *point_id = table_->iterator_get_id_(current_index_);
        }

        void get(int i, arma::vec *entry) {
          table_->iterator_get_(begin_ + i, entry);
        }

        void get_id(int i, int *point_id) {
          *point_id = table_->iterator_get_id_(begin_ + i);
        }

        void RandomPick(arma::vec *entry) {
          table_->iterator_get_(core::math::RandInt(begin_, end_), entry);
        }

        void RandomPick(arma::vec *entry, int *point_id) {
          int pre_translated_dfs_id = core::math::RandInt(begin_, end_);
          *point_id = table_->iterator_get_id_(pre_translated_dfs_id);
          table_->iterator_get_(pre_translated_dfs_id, entry);
        }

        void RandomPick(
          arma::vec *entry, int *point_id, double *weight) {
          int pre_translated_dfs_id = core::math::RandInt(begin_, end_);
          *point_id = table_->iterator_get_id_(pre_translated_dfs_id);
          table_->iterator_get_(pre_translated_dfs_id, entry, weight);
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

    void add_new_to_position_map(
      const std::map<int, int> &new_to_position_map_in) {
      new_to_position_map_ = new_to_position_map_in;
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
      SubTableType sub_table;
      TableType *this_table = const_cast<TableType *>(this);
      sub_table.Init(
        this_table, this_table->get_tree(), true);
      ar & sub_table;
    }

    /** @brief Unserialize the table.
     */
    template<class Archive>
    void load(Archive &ar, const unsigned int version) {
      SubTableType sub_table;
      sub_table.Init(
        this, this->get_tree(), true);
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
    const arma::mat &data() const {
      return data_;
    }

    /** @brief Returns a reference to the underlying data.
     */
    arma::mat &data() {
      return data_;
    }

    /** @brief Returns a reference to the weights associated with the
     *         underlying data.
     */
    const arma::mat &weights() const {
      return weights_;
    }

    /** @brief Returns a reference to the weights associated with the
     *         underlying data.
     */
    arma::mat &weights() {
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
      return data_.n_rows;
    }

    /** @brief Returns the number of points owned by the table.
     */
    int n_entries() const {
      return data_.n_cols;
    }

    /** @brief Initializes an empty table with the specified number of
     *         dimensions and number of points (optionally with the
     *         rank).
     */
    void Init(
      int num_dimensions_in, int num_points_in, int rank_in = 0) {
      rank_ = rank_in;
      if(num_dimensions_in > 0 && num_points_in > 0) {
        data_.zeros(num_dimensions_in, num_points_in);
        weights_.set_size(1, num_points_in);
        weights_.fill(1.0);
        if(core::table::global_m_file_) {
          old_from_new_ = core::table::global_m_file_->ConstructArray <
                          OldFromNewIndexType > (
                            data_.n_cols);
          new_from_old_ = core::table::global_m_file_->ConstructArray <
                          int > (
                            data_.n_cols);
        }
        else {
          old_from_new_ = new OldFromNewIndexType[data_.n_cols];
          new_from_old_ = new int[data_.n_cols];
        }
        core::tree::IndexInitializer<OldFromNewIndexType>::OldFromNew(
          data_, rank_in, old_from_new_.get());
      }
    }

    /** @brief Initializes the table from a file, optionally its rank.
     */
    void Init(
      const std::string &file_name,
      int rank_in = 0,
      const std::string *weight_file_name = NULL) {

      if(core::DatasetReader::ParseDataset(file_name, &data_) == false) {
        exit(0);
      }
      if(weight_file_name != NULL &&
          core::DatasetReader::ParseDataset(
            *weight_file_name, &weights_) == false) {
        exit(0);
      }
      if(weight_file_name == NULL) {
        if(weights_.n_rows == 0) {
          weights_.set_size(1, data_.n_cols);
        }
        weights_.fill(1.0);
      }
      rank_ = rank_in;

      if(core::table::global_m_file_) {
        old_from_new_ = core::table::global_m_file_->ConstructArray <
                        OldFromNewIndexType > (
                          data_.n_cols);
        new_from_old_ = core::table::global_m_file_->ConstructArray <
                        int > (
                          data_.n_cols);
      }
      else {
        if(old_from_new_ == NULL) {
          old_from_new_ = new OldFromNewIndexType[data_.n_cols];
        }
        if(new_from_old_ == NULL) {
          new_from_old_ = new int[data_.n_cols];
        }
      }
      core::tree::IndexInitializer<OldFromNewIndexType>::OldFromNew(
        data_, rank_in, old_from_new_.get());
    }

    /** @brief Saves the table to a text file.
     */
    void Save(
      const std::string &file_name,
      const std::string *weight_file_name = NULL) const {

      FILE *foutput = fopen(file_name.c_str(), "w+");
      FILE *woutput = (weight_file_name != NULL) ?
                      fopen(weight_file_name->c_str(), "w+") : NULL;
      for(unsigned int j = 0; j < data_.n_cols; j++) {
        arma::vec point;
        double point_weight;

        // Grab each point with its weight.
        this->direct_get_(j, &point, &point_weight);

        // Output the point.
        for(int i = 0; i < static_cast<int>(data_.n_rows); i++) {
          fprintf(foutput, "%g", point[i]);
          if(i < static_cast<int>(data_.n_rows) - 1) {
            fprintf(foutput, ",");
          }
        }
        fprintf(foutput, "\n");

        // Output the weight if requested.
        if(woutput != NULL) {
          fprintf(woutput, "%g\n", point_weight);
        }
      }
      fclose(foutput);
      if(woutput != NULL) {
        fclose(woutput);
      }
    }

    /** @brief Gets the frontier nodes of the indexed tree such that
     *         the subtree contains a specified number of points
     *         bounded by the parameter.
     */
    void get_frontier_nodes(
      int max_subtree_size,
      std::vector<TreeType *> *frontier_nodes_out) const {

      tree_->get_frontier_nodes(max_subtree_size, frontier_nodes_out);
    }

    /** @brief Gets the fronter nodes of the indexed tree up to a
     *         specified number of nodes.
     */
    void get_frontier_nodes_bounded_by_number(
      int max_num_nodes,
      std::vector <
      boost::intrusive_ptr<SubTableType> > *frontier_subtables_out) const {

      std::vector<TreeType *> frontier_nodes_out;
      tree_->get_frontier_nodes_bounded_by_number(
        max_num_nodes, &frontier_nodes_out);
      for(unsigned int i = 0; i < frontier_nodes_out.size(); i++) {
        frontier_subtables_out->push_back(
          boost::intrusive_ptr<SubTableType>(new SubTableType()));
        ((*frontier_subtables_out)[i])->Init(
          const_cast<TableType *>(this), frontier_nodes_out[i], false);
      }
    }

    /** @brief Builds the tree with a specified metric and a leaf
     *         size.
     */
    template<typename MetricType>
    void IndexData(
      const MetricType &metric_in, int leaf_size, int rank_in,
      int max_num_leaf_nodes = std::numeric_limits<int>::max()) {
      int num_nodes;
      rank_ = rank_in;
      tree_ = TreeType::MakeTree(
                metric_in, data_, weights_, leaf_size, old_from_new_.get(),
                new_from_old_.get(), max_num_leaf_nodes, &num_nodes, rank_);
    }

    /** @brief Gets the point with the specific ID.
     */
    void get(int point_id, arma::vec *point_out) const {
      direct_get_(point_id, point_out);
    }

    /** @brief Gets the point with the specific ID.
     */
    void get(int point_id, arma::vec *point_out) {
      direct_get_(point_id, point_out);
    }

    /** @brief Gets the point with the specific ID and its weight.
     */
    void get(
      int point_id, arma::vec *point_out, double *point_weight_out) const {
      direct_get_(point_id, point_out, point_weight_out);
    }

    /** @brief Gets the point with the specific ID and its weight.
     */
    void get(int point_id, arma::vec *point_out, double *point_weight_out) {
      direct_get_(point_id, point_out, point_weight_out);
    }

    /** @brief Prints the tree owned by the table.
     */
    void PrintTree() const {
      tree_->Print();
    }

    const double *GetColumnPtr(int point_id) const {
      if(this->IsIndexed() == false) {
        return core::table::GetColumnPtr(data_, point_id);
      }
      else {
        return core::table::GetColumnPtr(
                 data_,
                 IndexUtil<int>::Extract(
                   new_from_old_.get(), point_id));
      }
    }

  private:

    void direct_get_(
      int point_id, arma::vec *entry) const {
      if(this->IsIndexed() == false) {
        core::table::MakeColumnVector(data_, point_id, entry);
      }
      else {
        core::table::MakeColumnVector(
          data_,
          IndexUtil<int>::Extract(
            new_from_old_.get(), point_id), entry);
      }
    }

    void direct_get_(int point_id, arma::vec *entry) {
      if(this->IsIndexed() == false) {
        core::table::MakeColumnVector(data_, point_id, entry);
      }
      else {
        core::table::MakeColumnVector(
          data_,
          IndexUtil<int>::Extract(
            new_from_old_.get(), point_id), entry);
      }
    }

    void direct_get_(
      int point_id, arma::vec *entry, double *point_weight) const {
      if(this->IsIndexed() == false) {
        core::table::MakeColumnVector(data_, point_id, entry);
        *point_weight = weights_.at(0, point_id);
      }
      else {
        int located_position = IndexUtil<int>::Extract(
                                 new_from_old_.get(), point_id);
        core::table::MakeColumnVector(data_, located_position, entry);
        *point_weight = weights_.at(0, located_position);
      }
    }

    void direct_get_(int point_id, arma::vec *entry, double *point_weight) {
      if(this->IsIndexed() == false) {
        core::table::MakeColumnVector(data_, point_id, entry);
        *point_weight = weights_.at(0, point_id);
      }
      else {
        int located_position = IndexUtil<int>::Extract(
                                 new_from_old_.get(), point_id);
        core::table::MakeColumnVector(data_, located_position, entry);
        *point_weight = weights_.at(0, located_position);
      }
    }

    void iterator_get_(
      int reordered_position, arma::vec *entry) const {
      core::table::MakeColumnVector(
        data_, this->locate_reordered_position_(reordered_position), entry);
    }

    void iterator_get_(
      int reordered_position, arma::vec *entry, double *weight) const {
      int located_reordered_position =
        this->locate_reordered_position_(reordered_position);
      core::table::MakeColumnVector(data_, located_reordered_position, entry);
      *weight = weights_.at(0, located_reordered_position);
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
      std::map<int, int>::const_iterator it =
        new_to_position_map_.find(reordered_position);
      if(it != new_to_position_map_.end()) {
        return it->second;
      }
      else {
        return reordered_position;
      }
    }
};
}
}

#endif

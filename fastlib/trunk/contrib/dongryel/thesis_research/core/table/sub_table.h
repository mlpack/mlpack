/** @file sub_table.h
 *
 *  An abstraction to serialize a part of a dataset and its associated
 *  subtree.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_SUB_TABLE_H
#define CORE_TABLE_SUB_TABLE_H

#include <map>
#include <vector>
#include <boost/scoped_ptr.hpp>
#include <boost/intrusive_ptr.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/interprocess/offset_ptr.hpp>
#include <boost/utility.hpp>
#include "core/table/index_util.h"
#include "core/table/sub_dense_matrix.h"

namespace core {
namespace table {

extern MemoryMappedFile *global_m_file_;

template<typename IncomingTableType>
class SubTable;

template<typename IncomingTableType>
inline void intrusive_ptr_add_ref(
  core::table::SubTable<IncomingTableType> *ptr) {
  ptr->reference_count_++;
}

template<typename IncomingTableType>
inline void intrusive_ptr_release(
  core::table::SubTable<IncomingTableType> *ptr) {
  ptr->reference_count_--;
  if(ptr->reference_count_ == 0) {
    if(core::table::global_m_file_) {
      core::table::global_m_file_->DestroyPtr(ptr);
    }
    else {
      delete ptr;
    }
  }
}

/** @brief A subtable class for serializing/unserializing a part of a
 *         table object.
 */
template<typename IncomingTableType>
class SubTable {

  public:

    /** @brief The type of the table.
     */
    typedef IncomingTableType TableType;

    /** @brief The iterator type.
     */
    typedef typename TableType::TreeIterator TreeIteratorType;

    /** @brief The associated query result.
     */
    typedef typename TableType::QueryResultType QueryResultType;

    /** @brief The type of the tree.
     */
    typedef typename TableType::TreeType TreeType;

    /** @brief The type of the old from new indices.
     */
    typedef typename TableType::OldFromNewIndexType OldFromNewIndexType;

    /** @brief The type of the subtable.
     */
    typedef core::table::SubTable<TableType> SubTableType;

    /** @brief The ID type of the subtable.
     */
    typedef boost::tuple<int, int, int> SubTableIDType;

    /** @brief The class for indicating the node ID for which the
     *         points are serialized underneath.
     */
    class PointSerializeFlagType {
      private:

        // For boost serialization.
        friend class boost::serialization::access;

        /** @brief The begin index.
         */
        int begin_;

        /** @brief The count of the points.
         */
        int count_;

      public:

        /** @brief Returns the beginning index of the flag.
         */
        int begin() const {
          return begin_;
        }

        /** @brief Returns the count of the points.
         */
        int count() const {
          return count_;
        }

        /** @brief Returns the ending index of the flag.
         */
        int end() const {
          return begin_ + count_;
        }

        /** @brief Serialize/unserialize.
         */
        template<class Archive>
        void serialize(Archive &ar, const unsigned int version) {
          ar & begin_;
          ar & count_;
        }

        /** @brief The default constructor.
         */
        PointSerializeFlagType() {
          begin_ = 0;
          count_ = 0;
        }

        /** @brief Initialize with a begin/count pair.
         */
        PointSerializeFlagType(
          int begin_in, int count_in) {
          begin_ = begin_in;
          count_ = count_in;
        }
    };

  private:

    // For boost serialization.
    friend class boost::serialization::access;

    /** @brief The ID of the cache block the subtable is occupying.
     */
    int cache_block_id_;

    /** @brief The pointer to the underlying data.
     */
    arma::mat *data_;

    /** @brief Maps each DFS index to the position in the underlying
     *         dense matrix storing the data.
     */
    std::map<int, int> id_to_position_map_;

    /** @brief Whether the subtable is an alias of another subtable or
     *         not.
     */
    bool is_alias_;

    /** @brief The pointer to the new_from_old mapping.
     */
    boost::interprocess::offset_ptr<int> *new_from_old_;

    /** @brief The pointer to the old_from_new mapping.
     */
    boost::interprocess::offset_ptr<OldFromNewIndexType> *old_from_new_;

    /** @brief The rank of the MPI process from which every query
     *         subtable/query result is derived. If not equal to the
     *         current MPI process rank, these must be written back
     *         when the task queue runs out.
     */
    int originating_rank_;

    /** @brief Maps each position of the loaded underlying matrix to
     *         its original DFS index of the process from which it was
     *         received.
     */
    std::map<int, int> position_to_id_map_;

    /** @brief The associated query result. If not NULL, then this
     *         subtable is assumed to be a query subtable.
     */
    boost::scoped_ptr< QueryResultType > query_result_;

    /** @brief Whether to serialize the new from old mapping.
     */
    bool serialize_new_from_old_mapping_;

    /** @brief The list each terminal node that is being
     *         serialized/unserialized, the beginning index and its
     *         boolean flag whether the points under it are all
     *         serialized or not.
     */
    std::vector< PointSerializeFlagType > serialize_points_per_terminal_node_;

    /** @brief If not NULL, this points to the starting node whose
     *         subtree must be serialized.
     */
    TreeType *start_node_;

    /** @brief The table to be loaded/saved.
     */
    TableType *table_;

    /** @brief The pointer to the tree.
     */
    boost::interprocess::offset_ptr<TreeType> *tree_;

    /** @brief The pointer to the underlying weights.
     */
    arma::mat *weights_;

  public:

    /** @brief The number of reference count for this subtable.
     */
    long reference_count_;

  private:

    /** @brief Collects the tree nodes in a list form and marks
     *         whether each terminal node should have its points
     *         serialized or not.
     */
    void FillTreeNodes_(
      TreeType *node, int parent_node_index,
      std::vector< std::pair< TreeType *, int > > &sorted_nodes,
      int level) const {

      if(node != NULL) {
        sorted_nodes.push_back(
          std::pair<TreeType *, int>(node, parent_node_index));

        // If the node is not a leaf,
        if(node->is_leaf() == false) {
          int parent_node_index = sorted_nodes.size() - 1;
          FillTreeNodes_(
            node->left(), parent_node_index, sorted_nodes, level + 1);
          FillTreeNodes_(
            node->right(), parent_node_index, sorted_nodes, level + 1);
        }
      }
    }

  public:

    /** @brief Sets the incoming position to ID map.
     */
    void set_position_to_id_map(
      const std::map<int, int> &position_to_id_map_in) {
      position_to_id_map_ = position_to_id_map_in;
    }

    /** @brief Sets the incoming ID to position map.
     */
    void set_id_to_position_map(
      const std::map<int, int> &id_to_position_map_in) {
      id_to_position_map_ = id_to_position_map_in;
    }

    /** @brief Copies the query subtable to the current subtable.
     */
    void Copy(const SubTableType &source_subtable_in) {

      // Copy the query result.
      query_result_->Copy(* source_subtable_in.query_result());

      // Copy the query tree.
      TreeType *destination_start_node =
        start_node_->FindByBeginCount(
          source_subtable_in.start_node()->begin(),
          source_subtable_in.start_node()->count());
      std::vector< std::pair<TreeType * , TreeType * > > stack;
      stack.push_back(
        std::pair <
        TreeType * , TreeType * > (
          destination_start_node, source_subtable_in.start_node()));
      while(stack.size() > 0) {
        std::pair< TreeType *, TreeType *> destination_source_pair =
          stack.back();
        stack.pop_back();
        destination_source_pair.first->stat().summary_.Copy(
          destination_source_pair.second->stat().summary_);
        if(! destination_source_pair.first->is_leaf()) {
          std::pair< TreeType *, TreeType *> left_destination_source_pair(
            destination_source_pair.first->left(),
            destination_source_pair.second->left());
          std::pair< TreeType *, TreeType *> right_destination_source_pair(
            destination_source_pair.first->right(),
            destination_source_pair.second->right());
          stack.push_back(left_destination_source_pair);
          stack.push_back(right_destination_source_pair);
        }
      }
    }

    /** @brief Return true if the given subtable is included by the
     *         subtable.
     */
    bool includes(const SubTableType &test_subtable_in) const {
      SubTableIDType test_id = test_subtable_in.subtable_id();
      int test_end = test_id.get<1>() + test_id.get<2>();
      int this_end = start_node_->begin() + start_node_->count();
      return test_id.get<0>() == table_->rank() &&
             start_node_->begin() <= test_id.get<1>() && test_end <= this_end;
    }

    void set_query_result(const QueryResultType &query_result_in) {
      TreeIteratorType start_node_it =
        table_->get_node_iterator(start_node_);
      if(query_result_.get() == NULL) {
        boost::scoped_ptr<QueryResultType> tmp_result(new QueryResultType());
        query_result_.swap(tmp_result);
      }
      query_result_->Alias(query_result_in, start_node_it);
    }

    const std::map<int, int> &id_to_position_map() const {
      return id_to_position_map_;
    }

    const std::map<int, int> &position_to_id_map() const {
      return position_to_id_map_;
    }

    /** @brief Returns the identifier information of the
     *         subtable. Currently (rank, begin, count) is the ID.
     */
    SubTableIDType subtable_id() const {
      return SubTableIDType(
               table_->rank(), start_node_->begin(), start_node_->count());
    }

    /** @brief Sets the root node of the subtable.
     */
    void set_start_node(TreeType *start_node_in) {
      start_node_ = start_node_in;
      serialize_points_per_terminal_node_.resize(0);
      if(start_node_in == NULL) {
        serialize_points_per_terminal_node_.push_back(
          PointSerializeFlagType(0, data_->n_cols));
      }
      else {
        TreeIteratorType start_node_it =
          table_->get_node_iterator(start_node_in);
        serialize_points_per_terminal_node_.push_back(
          PointSerializeFlagType(
            start_node_in->begin(), start_node_in->count()));
        if(query_result_.get() != NULL) {
          query_result_->Alias(start_node_it);
        }
      }
    }

    bool serialize_new_from_old_mapping() const {
      return serialize_new_from_old_mapping_;
    }

    void set_cache_block_id(int cache_block_id_in) {
      cache_block_id_ = cache_block_id_in;
    }

    int cache_block_id() const {
      return cache_block_id_;
    }

    /** @brief Returns the list of terminal nodes for which the points
     *         underneath are available.
     */
    const std::vector <
    PointSerializeFlagType > &serialize_points_per_terminal_node() const {
      return serialize_points_per_terminal_node_;
    }

    /** @brief Manual destruction.
     */
    void Destruct() {
      if(is_alias_ == false && table_ != NULL) {
        if(core::table::global_m_file_) {
          core::table::global_m_file_->DestroyPtr(table_);
        }
        else {
          delete table_;
        }
      }
      is_alias_ = true;
      table_ = NULL;
      id_to_position_map_.clear();
      position_to_id_map_.clear();
    }

    /** @brief Returns whether the subtable is an alias of another
     *         subtable.
     */
    bool is_alias() const {
      return is_alias_;
    }

    /** @brief Returns the associated query result.
     */
    const QueryResultType *query_result() const {
      return query_result_.get();
    }

    /** @brief Returns the associated query result.
     */
    QueryResultType *query_result() {
      return query_result_.get();
    }

    /** @brief Returns whether the subtable is a query subtable.
     */
    bool is_query_subtable() const {
      return query_result_.get() != NULL ;
    }

    /** @brief Aliases another subtable.
     */
    void Alias(const SubTableType &subtable_in) {
      cache_block_id_ = subtable_in.cache_block_id();
      data_ = const_cast<SubTableType &>(subtable_in).data();
      id_to_position_map_ = subtable_in.id_to_position_map();
      is_alias_ = true;
      new_from_old_ = const_cast<SubTableType &>(subtable_in).new_from_old();
      old_from_new_ = const_cast<SubTableType &>(subtable_in).old_from_new();
      originating_rank_ = subtable_in.originating_rank();
      position_to_id_map_ = subtable_in.position_to_id_map();
      if(subtable_in.query_result() != NULL) {
        if(query_result_.get() == NULL) {
          boost::scoped_ptr<QueryResultType> tmp_result(new QueryResultType());
          query_result_.swap(tmp_result);
        }
        query_result_->Alias(*(subtable_in.query_result()));
      }
      serialize_new_from_old_mapping_ =
        subtable_in.serialize_new_from_old_mapping();
      serialize_points_per_terminal_node_ =
        subtable_in.serialize_points_per_terminal_node();
      start_node_ = const_cast< SubTableType &>(subtable_in).start_node();
      table_ = const_cast< SubTableType &>(subtable_in).table();
      tree_ = const_cast<SubTableType &>(subtable_in).tree();
      weights_ = const_cast<SubTableType &>(subtable_in).weights();
    }

    /** @brief Steals the ownership of the incoming subtable.
     */
    void operator=(const SubTableType &subtable_in) {
      cache_block_id_ = subtable_in.cache_block_id();
      data_ = const_cast<SubTableType &>(subtable_in).data();
      id_to_position_map_ = subtable_in.id_to_position_map();
      is_alias_ = subtable_in.is_alias();
      const_cast<SubTableType &>(subtable_in).is_alias_ = true;
      new_from_old_ = const_cast<SubTableType &>(subtable_in).new_from_old();
      old_from_new_ = const_cast<SubTableType &>(subtable_in).old_from_new();
      originating_rank_ = subtable_in.originating_rank();
      position_to_id_map_ = subtable_in.position_to_id_map();
      if(subtable_in.query_result() != NULL) {
        if(query_result_.get() == NULL) {
          boost::scoped_ptr<QueryResultType> tmp_result(new QueryResultType());
          query_result_.swap(tmp_result);
        }
        query_result_->Alias(*(subtable_in.query_result()));
      }
      serialize_new_from_old_mapping_ =
        subtable_in.serialize_new_from_old_mapping();
      serialize_points_per_terminal_node_ =
        subtable_in.serialize_points_per_terminal_node();
      table_ = const_cast< SubTableType &>(subtable_in).table();
      start_node_ = const_cast< SubTableType &>(subtable_in).start_node();
      weights_ = const_cast<SubTableType &>(subtable_in).weights();
      tree_ = const_cast<SubTableType &>(subtable_in).tree();
    }

    /** @brief Steals the ownership of the incoming subtable.
     */
    SubTable(const SubTableType &subtable_in) {
      reference_count_ = 0;
      this->operator=(subtable_in);
    }

    /** @brief Serialize the subtable.
     */
    template<class Archive>
    void save(Archive &ar, const unsigned int version) const {

      // Save the associated query result.
      bool is_query_subtable = (query_result_.get() != NULL);
      ar & is_query_subtable;
      if(is_query_subtable) {
        ar & (* query_result_);
      }

      // Save the rank.
      int rank = table_->rank();
      ar & rank;

      // Save the tree.
      int num_nodes = 0;
      std::vector< std::pair<TreeType *, int> > tree_nodes;
      FillTreeNodes_(start_node_, -1, tree_nodes, 0);
      num_nodes = tree_nodes.size();
      ar & num_nodes;
      for(unsigned int i = 0; i < tree_nodes.size(); i++) {
        ar & (*(tree_nodes[i].first));
        ar & tree_nodes[i].second;
      }

      // Save the matrix and the mappings if requested.
      {
        core::table::SubDenseMatrix<SubTableType> sub_data;
        core::table::SubDenseMatrix<SubTableType> sub_weights;

        // If the subtable is an alias, specify which subset to save.
        sub_data.Init(
          data_,
          &(const_cast <
            SubTableType * >(this)->serialize_points_per_terminal_node_),
          const_cast< std::map<int, int> & >(id_to_position_map_),
          const_cast< std::map<int, int> & >(position_to_id_map_));
        sub_weights.Init(
          weights_,
          &(const_cast <
            SubTableType * >(this)->serialize_points_per_terminal_node_),
          const_cast< std::map<int, int> & >(id_to_position_map_),
          const_cast< std::map<int, int> & >(position_to_id_map_));
        ar & sub_data;
        ar & sub_weights;

        // Direct mapping saving.
        core::table::IndexUtil<OldFromNewIndexType>::Serialize(
          ar, old_from_new_->get(),
          serialize_points_per_terminal_node_, id_to_position_map_, false);

        // Save whether the new from old mapping is going to be
        // serialized or not.
        ar & serialize_new_from_old_mapping_;
        if(serialize_new_from_old_mapping_) {
          core::table::IndexUtil<int>::Serialize(
            ar, new_from_old_->get(),
            serialize_points_per_terminal_node_, id_to_position_map_, false);
        }
      }
    }

    /** @brief Unserialize the subtable.
     */
    template<class Archive>
    void load(Archive &ar, const unsigned int version) {

      // Load the associated query result if available.
      bool is_query_subtable;
      ar & is_query_subtable;
      if(is_query_subtable) {
        if(query_result_.get() == NULL) {
          boost::scoped_ptr<QueryResultType> tmp_result(new QueryResultType());
          query_result_.swap(tmp_result);
        }
        ar & (* query_result_);
      }

      // Set the rank.
      int rank_in;
      ar & rank_in;
      table_->set_rank(rank_in);

      // Load up the max number of loads to receive.
      int num_nodes;
      ar & num_nodes;
      std::vector< std::pair<TreeType *, int> > tree_nodes(num_nodes);
      for(int i = 0; i < num_nodes; i++) {
        tree_nodes[i].first =
          (core::table::global_m_file_) ?
          core::table::global_m_file_->Construct<TreeType>() : new TreeType();
        ar & (*(tree_nodes[i].first));
        ar & tree_nodes[i].second;
      }

      // Do the pointer corrections, and have the tree point to the
      // 0-th element.
      for(unsigned int i = 1; i < tree_nodes.size(); i++) {
        int parent_node_index = tree_nodes[i].second;
        if(tree_nodes[parent_node_index].first->begin() ==
            tree_nodes[i].first->begin()) {
          tree_nodes[parent_node_index].first->set_left_child(
            (*data_), tree_nodes[i].first);
        }
        else {
          tree_nodes[parent_node_index].first->set_right_child(
            (*data_), tree_nodes[i].first);
        }
      }
      (*tree_) = tree_nodes[0].first;
      start_node_ = tree_nodes[0].first;

      // Load the data and the mappings if available.
      {
        core::table::SubDenseMatrix<SubTableType> sub_data;
        sub_data.Init(
          data_, &serialize_points_per_terminal_node_,
          id_to_position_map_, position_to_id_map_);
        ar & sub_data;
        core::table::SubDenseMatrix<SubTableType> sub_weights;
        sub_weights.Init(
          weights_,
          (std::vector< PointSerializeFlagType > *) NULL,
          id_to_position_map_, position_to_id_map_);
        ar & sub_weights;
        if(table_->mappings_are_aliased() == false) {
          (*old_from_new_) =
            (core::table::global_m_file_) ?
            core::table::global_m_file_->ConstructArray <
            OldFromNewIndexType > (data_->n_cols) :
            new OldFromNewIndexType[ data_->n_cols];
          (*new_from_old_) =
            (core::table::global_m_file_) ?
            core::table::global_m_file_->ConstructArray <
            int > (data_->n_cols) : new int[ data_->n_cols] ;
        }

        // Always serialize onto a consecutive block of memory to save
        // space.
        core::table::IndexUtil<OldFromNewIndexType>::Serialize(
          ar, old_from_new_->get(),
          serialize_points_per_terminal_node_, id_to_position_map_, true);

        // Find out whether the new from old mapping was serialized or
        // not, and load accordingly.
        ar & serialize_new_from_old_mapping_;
        if(serialize_new_from_old_mapping_) {
          core::table::IndexUtil<int>::Serialize(
            ar, new_from_old_->get(),
            serialize_points_per_terminal_node_, id_to_position_map_, true);
        }
      }

      // Load the node ids for which there are points underneath.
      table_->add_new_to_position_map(id_to_position_map_);
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    /** @brief Set the rank of the MPI process from which this
     *         subtable was received.
     */
    void set_originating_rank(int originating_rank_in) {
      originating_rank_ = originating_rank_in;
    }

    /** @brief The default constructor.
     */
    SubTable() {
      cache_block_id_ = 0;
      data_ = NULL;
      is_alias_ = true;
      new_from_old_ = NULL;
      old_from_new_ = NULL;
      originating_rank_ = -1;
      reference_count_ = 0;
      serialize_new_from_old_mapping_ = true;
      start_node_ = NULL;
      table_ = NULL;
      tree_ = NULL;
      weights_ = NULL;
    }

    /** @brief The destructor.
     */
    ~SubTable() {
      if(is_alias_ == false && table_ != NULL) {
        if(core::table::global_m_file_) {
          core::table::global_m_file_->DestroyPtr(table_);
        }
        else {
          delete table_;
        }
      }
    }

    /** @brief Returns the underlying table object.
     */
    TableType *table() const {
      return table_;
    }

    /** @brief Returns the starting node to be serialized.
     */
    TreeType *start_node() const {
      return start_node_;
    }

    /** @brief Returns the underlying weights.
     */
    arma::mat *weights() const {
      return weights_;
    }

    /** @brief Returns the underlying multi-dimensional data.
     */
    arma::mat *data() const {
      return data_;
    }

    int originating_rank() const {
      return originating_rank_;
    }

    /** @brief Returns the old_from_new mapping.
     */
    boost::interprocess::offset_ptr <
    OldFromNewIndexType > *old_from_new() const {
      return old_from_new_;
    }

    /** @brief Returns the new_from_old mapping.
     */
    boost::interprocess::offset_ptr<int> *new_from_old() const {
      return new_from_old_;
    }

    /** @brief Returns the tree owned by the subtable.
     */
    boost::interprocess::offset_ptr<TreeType> *tree() const {
      return tree_;
    }

    bool has_same_subtable_id(const std::pair<int, int> &sub_table_id) const {
      return start_node_->begin() == sub_table_id.first &&
             start_node_->count() == sub_table_id.second;
    }

    /** @brief Initializes a subtable before loading.
     */
    void Init(
      int cache_block_id_in,
      bool serialize_new_from_old_mapping_in) {

      // Set the cache block ID.
      cache_block_id_ = cache_block_id_in;

      // Allocate the table.
      table_ = (core::table::global_m_file_) ?
               core::table::global_m_file_->Construct<TableType>() :
               new TableType();

      // Finalize the intialization.
      this->Init(
        table_, (TreeType *) NULL, serialize_new_from_old_mapping_in);

      // Since table_ pointer is explicitly allocated, is_alias_ flag
      // is turned to false. It is important that it is here to
      // overwrite is_alias_ flag after ALL initializations are done.
      is_alias_ = false;
    }

    /** @brief Initializes a subtable from a pre-existing table before
     *         serializing a subset of it.
     */
    void Init(
      TableType *table_in, TreeType *start_node_in,
      bool serialize_new_from_old_mapping_in) {
      serialize_new_from_old_mapping_ = serialize_new_from_old_mapping_in;
      table_ = table_in;
      is_alias_ = true;
      originating_rank_ = table_->rank();
      data_ = &(table_in->data());
      this->set_start_node(start_node_in);
      weights_ = &(table_in->weights());
      old_from_new_ = table_in->old_from_new_offset_ptr();
      new_from_old_ = table_in->new_from_old_offset_ptr();
      tree_ = table_in->get_tree_offset_ptr();
    }
};
}
}

#endif

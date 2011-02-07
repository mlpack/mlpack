/** @file sub_table.h
 *
 *  An abstraction to serialize a part of a dataset and its associated
 *  subtree.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_SUB_TABLE_H
#define CORE_TABLE_SUB_TABLE_H

#include <vector>
#include <boost/serialization/serialization.hpp>
#include <boost/interprocess/offset_ptr.hpp>
#include <boost/utility.hpp>
#include "core/table/index_util.h"
#include "core/table/sub_dense_matrix.h"

namespace core {
namespace table {

extern MemoryMappedFile *global_m_file_;

/** @brief A subtable class for serializing/unserializing a part of a
 *         table object.
 */
template<typename IncomingTableType>
class SubTable {

  public:

    /** @brief The type of the table.
     */
    typedef IncomingTableType TableType;

    /** @brief The type of the tree.
     */
    typedef typename TableType::TreeType TreeType;

    /** @brief The type of the old from new indices.
     */
    typedef typename TableType::OldFromNewIndexType OldFromNewIndexType;

    /** @brief The type of the subtable.
     */
    typedef core::table::SubTable<TableType> SubTableType;

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

    /** @brief Whether to serialize the new from old mapping.
     */
    bool serialize_new_from_old_mapping_;

    /** @brief The ID of the cache block the subtable is occupying.
     */
    int cache_block_id_;

    /** @brief The table to be loaded/saved.
     */
    TableType *table_;

    /** @brief If not NULL, this points to the starting node whose
     *         subtree must be serialized.
     */
    TreeType *start_node_;

    /** @brief The maximum number of levels of subtree to
     *         serialize/unserialize.
     */
    int max_num_levels_to_serialize_;

    /** @brief The pointer to the underlying data.
     */
    core::table::DenseMatrix *data_;

    /** @brief The pointer to the old_from_new mapping.
     */
    boost::interprocess::offset_ptr<OldFromNewIndexType> *old_from_new_;

    /** @brief The pointer to the new_from_old mapping.
     */
    boost::interprocess::offset_ptr<int> *new_from_old_;

    /** @brief The pointer to the tree.
     */
    boost::interprocess::offset_ptr<TreeType> *tree_;

    /** @brief Whether the subtable is an alias of another subtable or
     *         not.
     */
    bool is_alias_;

    /** @brief The list each terminal node that is being
     *         serialized/unserialized, the beginning index and its
     *         boolean flag whether the points under it are all
     *         serialized or not.
     */
    std::vector< PointSerializeFlagType > serialize_points_per_terminal_node_;

  private:

    /** @brief Collects the tree nodes in a list form and marks
     *         whether each terminal node should have its points
     *         serialized or not.
     */
    void FillTreeNodes_(
      TreeType *node, int parent_node_index,
      std::vector< std::pair< TreeType *, int > > &sorted_nodes,
      std::vector <
      PointSerializeFlagType > *serialize_points_per_terminal_node_in,
      int level) const {

      if(node != NULL && level <= max_num_levels_to_serialize_) {
        sorted_nodes.push_back(
          std::pair<TreeType *, int>(node, parent_node_index));

        // If the node is not a leaf,
        if(node->is_leaf() == false) {

          // Recurse only if the level is still low.
          if(level < max_num_levels_to_serialize_) {
            int parent_node_index = sorted_nodes.size() - 1;
            FillTreeNodes_(
              node->left(), parent_node_index, sorted_nodes,
              serialize_points_per_terminal_node_in, level + 1);
            FillTreeNodes_(
              node->right(), parent_node_index, sorted_nodes,
              serialize_points_per_terminal_node_in, level + 1);
          }
        }

        // In case it is a leaf, we are always stuck, so we grab the
        // points belonging to it as well.
        else {
          serialize_points_per_terminal_node_in->push_back(
            PointSerializeFlagType(node->begin(), node->count()));
        }
      }
    }

  public:

    bool serialize_new_from_old_mapping() const {
      return serialize_new_from_old_mapping_;
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

    /** @brief Returns whether the subtable is an alias of another
     *         subtable.
     */
    bool is_alias() const {
      return is_alias_;
    }

    /** @brief Steals the ownership of the incoming subtable.
     */
    void operator=(const SubTable<TableType> &subtable_in) {
      serialize_new_from_old_mapping_ =
        subtable_in.serialize_new_from_old_mapping();
      cache_block_id_ = subtable_in.cache_block_id();
      table_ = const_cast< SubTableType &>(subtable_in).table();
      start_node_ = const_cast< SubTableType &>(subtable_in).start_node();
      max_num_levels_to_serialize_ =
        const_cast<SubTableType &>(subtable_in).max_num_levels_to_serialize();
      data_ = const_cast<SubTableType &>(subtable_in).data();
      old_from_new_ = const_cast<SubTableType &>(subtable_in).old_from_new();
      new_from_old_ = const_cast<SubTableType &>(subtable_in).new_from_old();
      tree_ = const_cast<SubTableType &>(subtable_in).tree();
      is_alias_ = subtable_in.is_alias();
      const_cast<SubTableType &>(subtable_in).is_alias_ = true;
      serialize_points_per_terminal_node_ =
        subtable_in.serialize_points_per_terminal_node();
    }

    /** @brief Steals the ownership of the incoming subtable.
     */
    SubTable(const SubTable<TableType> &subtable_in) {
      this->operator=(subtable_in);
    }

    /** @brief Serialize the subtable.
     */
    template<class Archive>
    void save(Archive &ar, const unsigned int version) const {

      // Save the rank.
      int rank = table_->rank();
      ar & rank;

      // Save the tree.
      int num_nodes = 0;
      std::vector< std::pair<TreeType *, int> > tree_nodes;
      std::vector< PointSerializeFlagType >
      &serialize_points_per_terminal_node_alias =
        const_cast< std::vector<PointSerializeFlagType> & >(
          serialize_points_per_terminal_node_);
      FillTreeNodes_(
        start_node_, -1, tree_nodes,
        &serialize_points_per_terminal_node_alias, 0);
      num_nodes = tree_nodes.size();
      ar & num_nodes;
      for(unsigned int i = 0; i < tree_nodes.size(); i++) {
        ar & (*(tree_nodes[i].first));
        ar & tree_nodes[i].second;
      }

      // Save the node ids for which there are points available
      // underneath.
      int serialize_points_per_terminal_node_size =
        static_cast<int>(serialize_points_per_terminal_node_.size());
      ar & serialize_points_per_terminal_node_size;
      for(unsigned int i = 0;
          i < serialize_points_per_terminal_node_.size(); i++) {
        ar & serialize_points_per_terminal_node_[i];
      }

      // Save the matrix and the mappings if requested.
      {
        core::table::SubDenseMatrix<SubTableType> sub_data;
        sub_data.Init(data_, serialize_points_per_terminal_node_);
        ar & sub_data;

        // Direct mapping saving.
        core::table::IndexUtil<OldFromNewIndexType>::Serialize(
          ar, old_from_new_->get(),
          serialize_points_per_terminal_node_, false);

        // Save whether the new from old mapping is going to be
        // serialized or not.
        ar & serialize_new_from_old_mapping_;
        if(serialize_new_from_old_mapping_) {
          core::table::IndexUtil<int>::Serialize(
            ar, new_from_old_->get(),
            serialize_points_per_terminal_node_, false);
        }
      }
    }

    /** @brief Unserialize the subtable.
     */
    template<class Archive>
    void load(Archive &ar, const unsigned int version) {

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

      // Load the node ids for which there are points underneath.
      int serialize_points_per_terminal_node_size;
      ar & serialize_points_per_terminal_node_size;
      serialize_points_per_terminal_node_.resize(
        serialize_points_per_terminal_node_size);
      for(int i = 0; i < serialize_points_per_terminal_node_size; i++) {
        ar & serialize_points_per_terminal_node_[i];

        // Add the list of points that are serialized to the table so
        // that the iterators work properly.
        table_->add_begin_count_pairs(
          serialize_points_per_terminal_node_[i].begin(),
          serialize_points_per_terminal_node_[i].count());
      }

      // Load the data and the mappings if available.
      {
        core::table::SubDenseMatrix<SubTableType> sub_data;
        sub_data.Init(data_, serialize_points_per_terminal_node_);
        ar & sub_data;
        if(table_->mappings_are_aliased() == false) {
          (*old_from_new_) =
            (core::table::global_m_file_) ?
            core::table::global_m_file_->ConstructArray <
            OldFromNewIndexType > (data_->n_cols()) :
            new OldFromNewIndexType[ data_->n_cols()];
          (*new_from_old_) =
            (core::table::global_m_file_) ?
            core::table::global_m_file_->ConstructArray <
            int > (data_->n_cols()) : new int[ data_->n_cols()] ;
        }

        // Always serialize onto a consecutive block of memory to save
        // space.
        core::table::IndexUtil<OldFromNewIndexType>::Serialize(
          ar, old_from_new_->get(),
          serialize_points_per_terminal_node_, true);

        // Find out whether the new from old mapping was serialized or
        // not, and load accordingly.
        ar & serialize_new_from_old_mapping_;
        if(serialize_new_from_old_mapping_) {
          core::table::IndexUtil<int>::Serialize(
            ar, new_from_old_->get(),
            serialize_points_per_terminal_node_, true);
        }
      }
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    /** @brief The default constructor.
     */
    SubTable() {
      serialize_new_from_old_mapping_ = true;
      cache_block_id_ = 0;
      table_ = NULL;
      start_node_ = NULL;
      max_num_levels_to_serialize_ = std::numeric_limits<int>::max();
      data_ = NULL;
      old_from_new_ = NULL;
      new_from_old_ = NULL;
      tree_ = NULL;
      is_alias_ = true;
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

    /** @brief Returns the maximum level underneath the starting node
     *         to be serialized/unserialized.
     */
    int max_num_levels_to_serialize() const {
      return max_num_levels_to_serialize_;
    }

    /** @brief Returns the underlying multi-dimensional data.
     */
    core::table::DenseMatrix *data() const {
      return data_;
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

    /** @brief Initializes a subtable before loading.
     */
    void Init(
      int cache_block_id_in,
      int max_num_levels_to_serialize_in,
      bool serialize_new_from_old_mapping_in) {

      // Set the cache block ID.
      cache_block_id_ = cache_block_id_in;

      // Allocate the table.
      table_ = (core::table::global_m_file_) ?
               core::table::global_m_file_->Construct<TableType>() :
               new TableType();

      // Finalize the intialization.
      this->Init(
        table_, (TreeType *) NULL, max_num_levels_to_serialize_in,
        serialize_new_from_old_mapping_in);

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
      int max_num_levels_to_serialize_in,
      bool serialize_new_from_old_mapping_in) {
      serialize_new_from_old_mapping_ = serialize_new_from_old_mapping_in;
      table_ = table_in;
      is_alias_ = true;
      start_node_ = start_node_in;
      max_num_levels_to_serialize_ = max_num_levels_to_serialize_in;
      data_ = &table_in->data();
      old_from_new_ = table_in->old_from_new_offset_ptr();
      new_from_old_ = table_in->new_from_old_offset_ptr();
      tree_ = table_in->get_tree_offset_ptr();
    }
};
}
}

#endif

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
template<typename TableType>
class SubTable {

  public:

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

    TableType *table_;

    TreeType *start_node_;

    int max_num_levels_to_serialize_;

    core::table::DenseMatrix *data_;

    boost::interprocess::offset_ptr<OldFromNewIndexType> *old_from_new_;

    boost::interprocess::offset_ptr<int> *new_from_old_;

    boost::interprocess::offset_ptr<TreeType> *tree_;

    bool is_alias_;

    /** @brief The list each terminal node that is being
     *         serialized/unserialized, the beginning index and its
     *         boolean flag whether the points under it are all
     *         serialized or not.
     */
    std::vector< PointSerializeFlagType > serialize_points_per_terminal_node_;

  private:

    void FillTreeNodes_(
      TreeType *node, int node_index,
      std::vector<TreeType *> &sorted_nodes, int *num_nodes,
      std::vector <
      PointSerializeFlagType > *serialize_points_per_terminal_node_in,
      int level) const {

      if(node != NULL && level <= max_num_levels_to_serialize_) {
        (*num_nodes)++;
        sorted_nodes[node_index] = node;

        if(node->is_leaf() == false) {

          // Recurse only if the level is still low.
          if(level < max_num_levels_to_serialize_) {
            FillTreeNodes_(
              node->left(), 2 * node_index + 1, sorted_nodes, num_nodes,
              serialize_points_per_terminal_node_in, level + 1);
            FillTreeNodes_(
              node->right(), 2 * node_index + 2, sorted_nodes, num_nodes,
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

    int FindTreeDepth_(TreeType *node, int level) const {
      if(node == NULL || level > max_num_levels_to_serialize_) {
        return 0;
      }
      int left_depth = FindTreeDepth_(node->left(), level + 1);
      int right_depth = FindTreeDepth_(node->right(), level + 1);
      return (left_depth > right_depth) ? (left_depth + 1) : (right_depth + 1);
    }

  public:

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
      int tree_depth = FindTreeDepth_(start_node_, 0);
      int max_size = 1 << tree_depth;
      std::vector< TreeType *> tree_nodes(max_size, (TreeType *) NULL);
      std::vector< PointSerializeFlagType >
      &serialize_points_per_terminal_node_alias =
        const_cast< std::vector<PointSerializeFlagType> & >(
          serialize_points_per_terminal_node_);
      FillTreeNodes_(
        start_node_, 0, tree_nodes, &num_nodes,
        &serialize_points_per_terminal_node_alias, 0);
      ar & max_size;
      ar & num_nodes;
      for(unsigned int i = 0; i < tree_nodes.size(); i++) {
        if(tree_nodes[i]) {
          ar & i;
          ar & (*(tree_nodes[i]));
        }
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
        core::table::IndexUtil<OldFromNewIndexType>::Serialize(
          ar, old_from_new_->get(), data_->n_cols(),
          serialize_points_per_terminal_node_);
        core::table::IndexUtil<int>::Serialize(
          ar, new_from_old_->get(), data_->n_cols(),
          serialize_points_per_terminal_node_);
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
      int max_num_nodes;
      int num_nodes;
      ar & max_num_nodes;
      ar & num_nodes;
      std::vector< TreeType *> tree_nodes(max_num_nodes, (TreeType *) NULL);
      for(int i = 0; i < num_nodes; i++) {
        int node_index;
        ar & node_index;
        tree_nodes[node_index] =
          (core::table::global_m_file_) ?
          core::table::global_m_file_->Construct<TreeType>() : new TreeType();
        ar & (*(tree_nodes[node_index]));
      }

      // Do the pointer corrections, and have the tree point to the
      // 0-th element.
      for(unsigned int i = 0; i < tree_nodes.size(); i++) {
        if(tree_nodes[i] && 2 * i + 2 < tree_nodes.size()) {
          tree_nodes[i]->set_children(
            (*data_), tree_nodes[2 * i + 1], tree_nodes[2 * i + 2]);
        }
      }
      (*tree_) = tree_nodes[0];
      start_node_ = tree_nodes[0];

      // Load the node ids for which there are points underneath.
      int serialize_points_per_terminal_node_size;
      ar & serialize_points_per_terminal_node_size;
      serialize_points_per_terminal_node_.resize(
        serialize_points_per_terminal_node_size);
      for(int i = 0; i < serialize_points_per_terminal_node_size; i++) {
        ar & serialize_points_per_terminal_node_[i];
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
        core::table::IndexUtil<OldFromNewIndexType>::Serialize(
          ar, old_from_new_->get(), data_->n_cols(),
          serialize_points_per_terminal_node_);
        core::table::IndexUtil<int>::Serialize(
          ar, new_from_old_->get(), data_->n_cols(),
          serialize_points_per_terminal_node_);
      }
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    /** @brief The default constructor.
     */
    SubTable() {
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

    core::table::DenseMatrix *data() const {
      return data_;
    }

    boost::interprocess::offset_ptr <
    OldFromNewIndexType > *old_from_new() const {
      return old_from_new_;
    }

    boost::interprocess::offset_ptr<int> *new_from_old() const {
      return new_from_old_;
    }

    boost::interprocess::offset_ptr<TreeType> *tree() const {
      return tree_;
    }

    template<typename OldFromNewIndexType>
    void Init(
      int rank_in, core::table::DenseMatrix &data_alias_in,
      OldFromNewIndexType *old_from_new_alias_in,
      int *new_from_old_alias_in,
      int max_num_levels_to_serialize_in) {
      table_ = (core::table::global_m_file_) ?
               core::table::global_m_file_->Construct<TableType>() :
               new TableType();

      // Make the data grab the pointer.
      table_->data().Alias(
        data_alias_in.ptr(), data_alias_in.n_rows(), data_alias_in.n_cols());
      table_->set_rank(rank_in);

      // Alias the incoming mappings.
      table_->Alias(old_from_new_alias_in, new_from_old_alias_in);

      // Finalize the intialization.
      this->Init(table_, (TreeType *) NULL, max_num_levels_to_serialize_in);

      // Since table_ pointer is explicitly allocated, is_alias_ flag
      // is turned to false. It is important that it is here to
      // overwrite is_alias_ flag after ALL initializations are done.
      is_alias_ = false;
    }

    void Init(
      TableType *table_in, TreeType *start_node_in,
      int max_num_levels_to_serialize_in) {
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

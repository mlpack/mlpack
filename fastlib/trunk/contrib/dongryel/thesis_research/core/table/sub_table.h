/** @file sub_table.h
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

template<typename TableType>
class SubTable {

  public:
    typedef typename TableType::TreeType TreeType;

    typedef typename TableType::OldFromNewIndexType OldFromNewIndexType;

    typedef SubTable<TableType> SubTableType;

  private:
    friend class boost::serialization::access;

    TableType *table_;

    TreeType *start_node_;

    int max_num_levels_to_serialize_;

    core::table::DenseMatrix *data_;

    boost::interprocess::offset_ptr<OldFromNewIndexType> *old_from_new_;

    boost::interprocess::offset_ptr<int> *new_from_old_;

    boost::interprocess::offset_ptr<TreeType> *tree_;

    bool is_alias_;

  private:

    void FillTreeNodes_(
      TreeType *node, int node_index, std::vector<TreeType *> &sorted_nodes,
      int *num_nodes,
      std::vector<bool> *serialize_points_per_terminal_node,
      int level) const {

      if(node != NULL && level <= max_num_levels_to_serialize_) {
        (*num_nodes)++;
        sorted_nodes[node_index] = node;

        if(node->is_leaf() == false) {
          FillTreeNodes_(
            node->left(), 2 * node_index + 1, sorted_nodes, num_nodes,
            serialize_points_per_terminal_node, level + 1);
          FillTreeNodes_(
            node->right(), 2 * node_index + 2, sorted_nodes, num_nodes,
            serialize_points_per_terminal_node, level + 1);
        }

        // In case it is a leaf, grab the points belonging to it as
        // well.
        else if(level < max_num_levels_to_serialize_) {
          (*serialize_points_per_terminal_node)[node_index] = true;
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

    void operator=(const SubTable<TableType> &subtable_in) {
      table_ = const_cast< SubTableType &>(subtable_in).table();
      start_node_ = const_cast< SubTableType &>(subtable_in).start_node();
      max_num_levels_to_serialize_ =
        const_cast<SubTableType &>(subtable_in).max_num_levels_to_serialize();
      data_ = const_cast<SubTableType &>(subtable_in).data();
      old_from_new_ = const_cast<SubTableType &>(subtable_in).old_from_new();
      new_from_old_ = const_cast<SubTableType &>(subtable_in).new_from_old();
      tree_ = const_cast<SubTableType &>(subtable_in).tree();
    }

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
      std::vector<bool> serialize_points_per_terminal_node(max_size, false);
      FillTreeNodes_(
        start_node_, 0, tree_nodes, &num_nodes,
        &serialize_points_per_terminal_node, 0);
      ar & max_size;
      ar & num_nodes;
      for(unsigned int i = 0; i < tree_nodes.size(); i++) {
        if(tree_nodes[i]) {
          ar & i;
          ar & (*(tree_nodes[i]));
        }
      }

      // Save the boolean flags.
      ar & serialize_points_per_terminal_node;

      // Save the matrix and the mappings if requested.
      {
        core::table::SubDenseMatrix<TreeType> sub_data;
        sub_data.Init(data_, tree_nodes, serialize_points_per_terminal_node);
        ar & sub_data;
        core::table::IndexUtil<OldFromNewIndexType>::Serialize(
          ar, old_from_new_->get(), data_->n_cols(),
          tree_nodes, serialize_points_per_terminal_node);
        core::table::IndexUtil<int>::Serialize(
          ar, new_from_old_->get(), data_->n_cols(),
          tree_nodes, serialize_points_per_terminal_node);
      }
    }

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

      // Load the boolean flags.
      std::vector<bool> serialize_points_per_terminal_node;
      ar & serialize_points_per_terminal_node;

      // Load the data and the mappings if available.
      {
        core::table::SubDenseMatrix<TreeType> sub_data;
        sub_data.Init(data_, tree_nodes, serialize_points_per_terminal_node);
        ar & sub_data;
        core::table::IndexUtil<OldFromNewIndexType>::Serialize(
          ar, old_from_new_->get(), data_->n_cols(),
          tree_nodes, serialize_points_per_terminal_node);
        core::table::IndexUtil<int>::Serialize(
          ar, new_from_old_->get(), data_->n_cols(),
          tree_nodes, serialize_points_per_terminal_node);
      }
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

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

    ~SubTable() {
      if(is_alias_ == false) {
        if(core::table::global_m_file_) {
          core::table::global_m_file_->DestroyPtr(table_);
        }
        else {
          delete table_;
        }
      }
    }

    TableType *table() const {
      return table_;
    }

    TreeType *start_node() const {
      return start_node_;
    }

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

    void Init(
      int rank_in, core::table::DenseMatrix &data_alias_in,
      int max_num_levels_to_serialize_in) {
      table_ = (core::table::global_m_file_) ?
               core::table::global_m_file_->Construct<TableType>() :
               new TableType();
      table_->data().Alias(
        data_alias_in.ptr(), data_alias_in.n_rows(), data_alias_in.n_cols());
      is_alias_ = false;
      table_->set_rank(rank_in);
      this->Init(table_, (TreeType *) NULL, max_num_levels_to_serialize_in);
    }

    void Init(
      TableType *table_in, TreeType *start_node_in,
      int max_num_levels_to_serialize_in) {
      table_ = table_in;
      start_node_ = start_node_in;
      max_num_levels_to_serialize_ = max_num_levels_to_serialize_in;
      data_ = &table_in->data();
      old_from_new_ = table_in->old_from_new_offset_ptr();
      new_from_old_ = table_in->new_from_old_offset_ptr();
      tree_ = table_in->get_tree_offset_ptr();
    }
};
};
};

#endif

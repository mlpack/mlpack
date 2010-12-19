/** @file sub_table.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_SUB_TABLE_H
#define CORE_TABLE_SUB_TABLE_H

#include <boost/serialization/serialization.hpp>
#include <boost/interprocess/offset_ptr.hpp>
#include <boost/utility.hpp>

#include "core/table/index_util.h"

namespace core {
namespace table {

extern MemoryMappedFile *global_m_file_;

template<typename TableType>
class SubTable: public boost::noncopyable {

  public:
    typedef typename TableType::TreeType TreeType;

    typedef typename TableType::OldFromNewIndexType OldFromNewIndexType;

  private:
    friend class boost::serialization::access;

    TableType *table_;

    TreeType *start_node_;

    int max_num_levels_to_serialize_;

    bool serialize_points_;

    core::table::DenseMatrix *data_;

    boost::interprocess::offset_ptr<OldFromNewIndexType> *old_from_new_;

    boost::interprocess::offset_ptr<int> *new_from_old_;

    boost::interprocess::offset_ptr<TreeType> *tree_;

  private:

    void FillTreeNodes_(
      TreeType *node, int node_index, std::vector<TreeType *> &sorted_nodes,
      int *num_nodes, int level) const {

      if(node != NULL && level <= max_num_levels_to_serialize_) {
        (*num_nodes)++;
        sorted_nodes[node_index] = node;

        if(node->is_leaf() == false) {
          FillTreeNodes_(
            node->left(), 2 * node_index + 1, sorted_nodes, num_nodes,
            level + 1);
          FillTreeNodes_(
            node->right(), 2 * node_index + 2, sorted_nodes, num_nodes,
            level + 1);
        }
      }
    }

    int FindTreeDepth_(TreeType *node, int level) const {
      if(node == NULL || level >= max_num_levels_to_serialize_) {
        return 0;
      }
      int left_depth = FindTreeDepth_(node->left(), level + 1);
      int right_depth = FindTreeDepth_(node->right(), level + 1);
      return (left_depth > right_depth) ? (left_depth + 1) : (right_depth + 1);
    }

  public:

    template<class Archive>
    void save(Archive &ar, const unsigned int version) const {

      // Save the flag whether the serialization of data were requested.
      ar & serialize_points_;

      // Save the matrix and the mappings if requested.
      if(serialize_points_) {
        ar & (*data_);
        core::table::IndexUtil<OldFromNewIndexType>::Serialize(
          ar, old_from_new_->get(), data_->n_cols());
        core::table::IndexUtil<int>::Serialize(
          ar, new_from_old_->get(), data_->n_cols());
      }

      // Save the rank.
      int rank = table_->rank();
      ar & rank;

      // Save the tree.
      int num_nodes = 0;
      int tree_depth = FindTreeDepth_(start_node_, 0);
      std::vector< TreeType *> tree_nodes(1 << tree_depth, (TreeType *) NULL);

      FillTreeNodes_(start_node_, 0, tree_nodes, &num_nodes, 0);
      int max_size = tree_nodes.size();
      ar & max_size;
      ar & num_nodes;
      for(unsigned int i = 0; i < tree_nodes.size(); i++) {
        if(tree_nodes[i]) {
          ar & i;
          ar & (*(tree_nodes[i]));
        }
      }
    }

    template<class Archive>
    void load(Archive &ar, const unsigned int version) {

      // Initialize so that everything gets loaded correctly.
      this->Init(table_, (TreeType *) NULL, std::numeric_limits<int>::max());

      // Load the point serialization flags.
      ar & serialize_points_;

      // Load the data and the mappings if available.
      if(serialize_points_) {
        ar & (*data_);
        (*old_from_new_) = (core::table::global_m_file_) ?
                           core::table::global_m_file_->ConstructArray <
                           OldFromNewIndexType > (data_->n_cols()) :
                           new OldFromNewIndexType[ data_->n_cols()];
        (*new_from_old_) = (core::table::global_m_file_) ?
                           core::table::global_m_file_->ConstructArray <
                           int > (data_->n_cols()) :
                           new int[ data_->n_cols()];
        core::table::IndexUtil<OldFromNewIndexType>::Serialize(
          ar, old_from_new_->get(), data_->n_cols());
        core::table::IndexUtil<int>::Serialize(
          ar, new_from_old_->get(), data_->n_cols());
      }

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
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    SubTable() {
      table_ = NULL;
      start_node_ = NULL;
      max_num_levels_to_serialize_ = std::numeric_limits<int>::max();
      serialize_points_ = true;
      data_ = NULL;
      old_from_new_ = NULL;
      new_from_old_ = NULL;
      tree_ = NULL;
    }

    bool serialize_points() const {
      return serialize_points_;
    }

    void set_serialize_points(bool flag_in) {
      serialize_points_ = flag_in;
    }

    void Init(
      TableType *table_in, TreeType *start_node_in,
      int max_num_levels_to_serialize_in) {
      table_ = table_in;
      start_node_ = start_node_in;
      max_num_levels_to_serialize_ = max_num_levels_to_serialize_in;
      serialize_points_ = true;
      data_ = &table_in->data();
      old_from_new_ = table_in->old_from_new_offset_ptr();
      new_from_old_ = table_in->new_from_old_offset_ptr();
      tree_ = table_in->get_tree_offset_ptr();
    }
};
};
};

#endif

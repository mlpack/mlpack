/** @file sub_table.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_SUB_TABLE_H
#define CORE_TABLE_SUB_TABLE_H

#include <boost/serialization/serialization.hpp>
#include <boost/interprocess/offset_ptr.hpp>
#include <boost/utility.hpp>

namespace core {
namespace table {

extern MemoryMappedFile *global_m_file_;

template<typename TableType>
class SubTable: public boost::noncopyable {

  public:
    typedef typename TableType::TreeType TreeType;

  private:
    friend class boost::serialization::access;

    TableType *table_;

    TreeType *root_node_;

    int level_;

    bool serialize_points_;

  public:

    template<class Archive>
    void save(Archive &ar, const unsigned int version) const {

      // Save the matrix and the rank.
      ar & data_;
      ar & rank_;

      // Save the old_from_new_mapping manually.
      for(int i = 0; i < data_.n_cols(); i++) {
        core::table::IndexUtil<OldFromNewIndexType>::Serialize(
          ar, old_from_new_.get(), i);
      }
      for(int i = 0; i < data_.n_cols(); i++) {
        core::table::IndexUtil<int>::Serialize(
          ar, new_from_old_.get(), i);
      }

      // Save the tree.
      int num_nodes = 0;
      int tree_depth = FindTreeDepth_(tree_.get());
      std::vector< TreeType *> tree_nodes(1 << tree_depth,  NULL);

      FillTreeNodes_(tree_.get(), 0, tree_nodes, &num_nodes);
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

      // Load the matrix and the rank.
      ar & data_;
      ar & rank_;

      // Load the mappings manually.
      old_from_new_ = (core::table::global_m_file_) ?
                      core::table::global_m_file_->ConstructArray <
                      OldFromNewIndexType > (data_.n_cols()) :
                      new OldFromNewIndexType[ data_.n_cols()];
      new_from_old_ = (core::table::global_m_file_) ?
                      core::table::global_m_file_->ConstructArray <
                      int > (data_.n_cols()) :
                      new int[ data_.n_cols()];
      for(int i = 0; i < data_.n_cols(); i++) {
        core::table::IndexUtil<OldFromNewIndexType>::Serialize(
          ar, old_from_new_.get(), i);
      }
      for(int i = 0; i < data_.n_cols(); i++) {
        core::table::IndexUtil<int>::Serialize(
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
        if(tree_nodes[i] && 2 * i + 2 < tree_nodes.size()) {
          tree_nodes[i]->set_children(
            data_, tree_nodes[2 * i + 1], tree_nodes[2 * i + 2]);
        }
      }
      tree_ = tree_nodes[0];
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    SubTable() {
      table_ = NULL;
      serialize_points_ = false;
    }

    void Init(TableType *table_in, TreeType *start_node_in, int level) {

    }
};
};
};

#endif

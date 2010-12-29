/** @file sub_table_list.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_SUB_TABLE_LIST_H
#define CORE_TABLE_SUB_TABLE_LIST_H

#include <boost/serialization/serialization.hpp>
#include <vector>

namespace core {
namespace table {
template<typename SubTableType>
class SubTableList {
  private:
    friend class boost::serialization::access;

    std::vector<SubTableType> list_;

  public:

    SubTableType &operator[](int pos) {
      return list_[pos];
    }

    unsigned int size() const {
      return list_.size();
    }

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      for(unsigned int i = 0; i < list_.size(); i++) {
        ar & list_[i];
      }
    }

    template<typename OldFromNewIndexType>
    void push_back(
      int rank_in, core::table::DenseMatrix &data_alias_in,
      OldFromNewIndexType *old_from_new_alias_in,
      int *new_from_old_alias_in,
      int max_num_levels_to_serialize_in) {
      list_.resize(list_.size() + 1);
      list_[list_.size() - 1].Init(
        rank_in, data_alias_in, old_from_new_alias_in, new_from_old_alias_in,
        max_num_levels_to_serialize_in);
    }

    template<typename TableType, typename TreeType>
    void push_back(
      TableType *table_in, TreeType *start_node_in,
      int max_num_levels_to_serialize_in) {
      list_.resize(list_.size() + 1);
      list_[list_.size() - 1].Init(
        table_in, start_node_in, max_num_levels_to_serialize_in);
    }
};
};
};

#endif

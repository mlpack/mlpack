/** @file sub_table_list.h
 *
 *  An abstract class to maintain a list of subtables to aid in the
 *  all-to-all exchange.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_SUB_TABLE_LIST_H
#define CORE_TABLE_SUB_TABLE_LIST_H

#include <boost/serialization/serialization.hpp>
#include <vector>
#include "core/table/dense_matrix.h"

namespace core {
namespace table {

/** @brief An abstract class for a list of subtables.
 */
template<typename IncomingSubTableType>
class SubTableList {
  public:

    /** @brief Convenient typedef of incoming subtable type so that it
     *         can be exported.
     */
    typedef IncomingSubTableType SubTableType;

  private:

    // For boost serialization.
    friend class boost::serialization::access;

    /** @brief The list of subtables.
     */
    std::vector<SubTableType> list_;

  public:

    /** @brief Resets the subtable list to be an empty list.
     */
    void Reset() {
      list_.resize(0);
    }

    /** @brief Returns the subtable at a given position.
     */
    const SubTableType &operator[](int pos) const {
      return list_[pos];
    }

    /** @brief Returns the subtable at a given position.
     */
    SubTableType &operator[](int pos) {
      return list_[pos];
    }

    /** @brief The size of the subtable list.
     */
    unsigned int size() const {
      return list_.size();
    }

    /** @brief Serializes each element of the subtable list.
     */
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      for(unsigned int i = 0; i < list_.size(); i++) {
        ar & list_[i];
      }
    }

    /** @brief Appends a subtable list.
     */
    template<typename SubTableListType>
    void push_back(const SubTableListType &list_in) {
      for(unsigned int i = 0; i < list_in.size(); i++) {
        list_.push_back(list_in[i]);
      }
    }

    /** @brief Pushes back a subtable for loading.
     */
    template<typename OldFromNewIndexType>
    void push_back(
      int rank_in, core::table::DenseMatrix &data_alias_in,
      OldFromNewIndexType *old_from_new_alias_in,
      int cache_block_id_in,
      int cache_block_size_in,
      int max_num_levels_to_serialize_in,
      bool serialize_new_from_old_mapping_in) {
      list_.resize(list_.size() + 1);
      list_[list_.size() - 1].Init(
        rank_in, data_alias_in, old_from_new_alias_in,
        cache_block_id_in, cache_block_size_in, max_num_levels_to_serialize_in,
        serialize_new_from_old_mapping_in);
    }

    /** @brief Pushes back a starting node and the number of levels to
     *         serialize under.
     */
    template<typename TableType, typename TreeType>
    void push_back(
      TableType *table_in, TreeType *start_node_in,
      int max_num_levels_to_serialize_in,
      bool serialize_new_from_old_mapping_in) {
      list_.resize(list_.size() + 1);
      list_[list_.size() - 1].Init(
        table_in, start_node_in, max_num_levels_to_serialize_in,
        serialize_new_from_old_mapping_in);
    }
};
}
}

#endif

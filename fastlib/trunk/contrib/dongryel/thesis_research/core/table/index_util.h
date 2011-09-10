/** @file index_util.h
 *
 *  A set of utilities for serializing/unserializing indices for
 *  maintaining the mapping order in shuffled points in a table.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_INDEX_UTIL_H
#define CORE_TABLE_INDEX_UTIL_H

namespace core {
namespace table {

/** @brief A template class for serializing an array of indices.
 */
template<typename IndexType>
class IndexUtil {
  public:

    /** @brief Serialize an index element at a given position.
     */
    static int Extract(IndexType *array, int position);

    /** @brief Serialize sets of specified index elements.
     */
    template<typename Archive, typename PointSerializeFlagArrayType>
    static void Serialize(
      Archive &ar, IndexType *array,
      const PointSerializeFlagArrayType &serialize_points_per_terminal_node,
      const std::map<int, int> &id_to_position_map, bool is_load_mode);
};

/** @brief A template specialization of the IndexUtil class for an int
 *         type.
 */
template<>
class IndexUtil< int > {
  public:
    static int Extract(int *array, int position) {
      return array[position];
    }

    template<typename Archive, typename PointSerializeFlagArrayType>
    static void Serialize(
      Archive &ar, int *array,
      const PointSerializeFlagArrayType &serialize_points_per_terminal_node,
      const std::map<int, int> &id_to_position_map, bool is_load_mode) {

      // Serialize onto a consecutive block.
      int index = 0;
      for(unsigned int j = 0;
          j < serialize_points_per_terminal_node.size(); j++) {
        index = (is_load_mode) ?
                index : serialize_points_per_terminal_node[j].begin();
        if(! is_load_mode) {
          typename std::map<int, int>::const_iterator it =
            id_to_position_map.find(index);
          if(it != id_to_position_map.end()) {
            index = it->second;
          }
        }
        for(int i = serialize_points_per_terminal_node[j].begin();
            i < serialize_points_per_terminal_node[j].end(); i++, index++) {
          ar & array[index];
        }
      }
    }
};

/** @brief A template specialization of the IndexUtil class for the
 *         distributed table old_from_new index mappings.
 */
template<>
class IndexUtil< std::pair<int, std::pair<int, int> > > {
  public:
    static int Extract(
      std::pair<int, std::pair<int, int> > *array, int position) {
      return array[position].second.second;
    }

    template<typename Archive, typename PointSerializeFlagArrayType>
    static void Serialize(
      Archive &ar, std::pair<int, std::pair<int, int> > *array,
      const PointSerializeFlagArrayType &serialize_points_per_terminal_node,
      const std::map<int, int> &id_to_position_map, bool is_load_mode) {

      int index = 0;
      for(unsigned int j = 0;
          j < serialize_points_per_terminal_node.size(); j++) {
        index = (is_load_mode) ?
                index : serialize_points_per_terminal_node[j].begin();
        if(! is_load_mode) {
          typename std::map<int, int>::const_iterator it =
            id_to_position_map.find(index);
          if(it != id_to_position_map.end()) {
            index = it->second;
          }
        }
        for(int i = serialize_points_per_terminal_node[j].begin();
            i < serialize_points_per_terminal_node[j].end(); i++, index++) {
          ar & array[index].first;
          ar & array[index].second.first;
          ar & array[index].second.second;
        }
      }
    }
};
}
}

#endif

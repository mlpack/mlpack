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

    /** @brief Serialize a specified number of index elements from a
     *         set of index elements.
     */
    template<typename Archive>
    static void Serialize(Archive &ar, IndexType *array, int num_elements);

    /** @brief Serialize sets of specified index elements.
     */
    template<typename Archive, typename PointSerializeFlagArrayType>
    static void Serialize(
      Archive &ar, IndexType *array, int num_elements,
      const PointSerializeFlagArrayType &serialize_points_per_terminal_node);
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

    template<typename Archive>
    static void Serialize(Archive &ar, int *array, int num_elements) {
      if(array == NULL) {
        return;
      }
      for(int i = 0; i < num_elements; i++) {
        ar & array[i];
      }
    }

    template<typename Archive, typename PointSerializeFlagArrayType>
    static void Serialize(
      Archive &ar, int *array, int num_elements,
      const PointSerializeFlagArrayType &serialize_points_per_terminal_node) {
      if(array == NULL) {
        return;
      }
      for(unsigned int j = 0;
          j < serialize_points_per_terminal_node.size(); j++) {
        for(int i = serialize_points_per_terminal_node[j].begin();
            i < serialize_points_per_terminal_node[j].end(); i++) {
          ar & array[i];
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

    template<typename Archive>
    static void Serialize(
      Archive &ar, std::pair<int, std::pair<int, int> > *array,
      int num_elements) {
      if(array == NULL) {
        return;
      }
      for(int i = 0; i < num_elements; i++) {
        ar & array[i].first;
        ar & array[i].second.first;
        ar & array[i].second.second;
      }
    }

    template<typename Archive, typename PointSerializeFlagArrayType>
    static void Serialize(
      Archive &ar, std::pair<int, std::pair<int, int> > *array,
      int num_elements,
      const PointSerializeFlagArrayType &serialize_points_per_terminal_node) {
      if(array == NULL) {
        return;
      }
      for(unsigned int j = 0;
          j < serialize_points_per_terminal_node.size(); j++) {
        for(int i = serialize_points_per_terminal_node[j].begin();
            i < serialize_points_per_terminal_node[j].end(); i++) {
          ar & array[i].first;
          ar & array[i].second.first;
          ar & array[i].second.second;
        }
      }
    }
};
}
}

#endif

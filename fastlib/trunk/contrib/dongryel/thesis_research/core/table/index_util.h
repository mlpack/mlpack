/** @file index_util.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_INDEX_UTIL_H
#define CORE_TABLE_INDEX_UTIL_H

namespace core {
namespace table {
template<typename IndexType>
class IndexUtil {
  public:
    static int Extract(IndexType *array, int position);

    template<typename Archive>
    static void Serialize(Archive &ar, IndexType *array, int num_elements);

    template<typename Archive, typename PointSerializeFlagArrayType>
    static void Serialize(
      Archive &ar, IndexType *array, int num_elements,
      const PointSerializeFlagArrayType &serialize_points_per_terminal_node);
};

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
        for(int i = serialize_points_per_terminal_node[j].begin_;
            i < serialize_points_per_terminal_node[j].end(); i++) {
          ar & array[i];
        }
      }
    }
};

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
        for(int i = serialize_points_per_terminal_node[j].begin_;
            i < serialize_points_per_terminal_node[j].end(); i++) {
          ar & array[i].first;
          ar & array[i].second.first;
          ar & array[i].second.second;
        }
      }
    }
};
};
};

#endif

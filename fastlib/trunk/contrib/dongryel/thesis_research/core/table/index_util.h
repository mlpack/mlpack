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
    static void Serialize(Archive &ar, IndexType *array, int position);
};

template<>
class IndexUtil< int > {
  public:
    static int Extract(int *array, int position) {
      return array[position];
    }

    template<typename Archive>
    static void Serialize(Archive &ar, int *array, int position) {
      ar & array[position];
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
      Archive &ar, std::pair<int, std::pair<int, int> > *array, int position) {
      ar & array[position].first;
      ar & array[position].second.first;
      ar & array[position].second.second;
    }
};
};
};

#endif

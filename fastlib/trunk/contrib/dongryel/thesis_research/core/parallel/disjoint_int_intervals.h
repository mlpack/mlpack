/** @file disjoint_int_intervals.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_DISJOINT_INT_INTERVALS_H
#define CORE_PARALLEL_DISJOINT_INT_INTERVALS_H

#include <map>

namespace core {
namespace parallel {

class DisjointIntIntervals {

  public:

    typedef std::pair<int, int> ValueType;

    typedef ValueType KeyType;

    struct ComparatorType {
      bool operator()(const KeyType &k1, const KeyType &k2) const {
        if(k1.second < k2.first) {
          return -1;
        }
        else if(k2.second < k1.first) {
          return 1;
        }
        else {
          return 0;
        }
      }
    };

    typedef std::map< KeyType, ValueType, ComparatorType > MapType;

  private:

    MapType intervals_;

  private:

    void Merge_(const ValueType &combine_with, ValueType *merged) {
      if(merged->first == combine_with.second) {
        merged->first = combine_with.first;
      }
      else {
        merged->second = combine_with.second;
      }
    }

  public:

    void Init() {
      intervals_.clear();
    }

    /** @brief Returns true if the test interval is not among the ones
     *         already available.
     */
    bool Insert(const ValueType &test_interval) {
      std::pair <
      typename MapType::iterator,
               typename MapType::iterator > intersecting_list =
                 intervals_.equal_range(test_interval);
      bool does_not_exist = true;
      if(intersecting_list.first != intersecting_list.second) {

        // Merge with every interval that intersects the incoming one.
        typename MapType::iterator current_it = intersecting_list.first;
        std::pair<int, int> merged = test_interval;
        do {
          if(current_it->first.first <= test_interval.first &&
              test_interval.second <= current_it->first.second) {
            does_not_exist = false;
          }
          Merge_(current_it->first, &merged);
          typename MapType::iterator next_it = current_it;
          next_it++;
          intervals_.erase(current_it);
          current_it = next_it;
        }
        while(current_it != intersecting_list.second);
        intervals_.insert(std::pair< KeyType, ValueType>(merged, merged));
      }
      else {

        // Otherwise, insert the incoming one.
        intervals_.insert(
          std::pair<KeyType, ValueType>(test_interval, test_interval));
      }
      return does_not_exist;
    }
};
}
}

#endif

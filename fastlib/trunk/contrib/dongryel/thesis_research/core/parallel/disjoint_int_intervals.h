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

    MapType *intervals_;

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

    const MapType &interval(int index) const {
      return intervals_[index];
    }

    ~DisjointIntIntervals() {
      delete[] intervals_;
    }

    DisjointIntIntervals() {
      intervals_ = NULL;
    }

    void Init(boost::mpi::communicator &world) {
      intervals_ = new MapType[ world.size()] ;
    }

    DisjointIntIntervals(
      boost::mpi::communicator &world,
      const DisjointIntIntervals &intervals_in) {
      intervals_ = new MapType[ world.size()];
      for(int i = 0; i < world.size(); i++) {
        intervals_[i] = intervals_in.interval(i);
      }
    }

    /** @brief Returns true if the test interval is not among the ones
     *         already available.
     */
    bool Insert(const boost::tuple<int, int, int> &interval_in) {
      bool does_not_exist = true;
      int rank = interval_in.get<0>();
      std::pair<int, int> test_interval(
        interval_in.get<1>(), interval_in.get<2>());
      if(intervals_[rank].size() > 0) {
        std::pair <
        MapType::iterator,
                MapType::iterator > intersecting_list =
                  intervals_[rank].equal_range(test_interval);
        if(intersecting_list.first != intersecting_list.second) {

          // Merge with every interval that intersects the incoming one.
          MapType::iterator current_it = intersecting_list.first;
          std::pair<int, int> merged(
            test_interval.first, test_interval.second);
          do {
            if(current_it->first.first <= test_interval.first &&
                test_interval.second <= current_it->first.second) {
              does_not_exist = false;
            }
            Merge_(current_it->first, &merged);
            current_it++;
          }
          while(current_it != intersecting_list.second);
          intervals_[rank].erase(
            intersecting_list.first, intersecting_list.second);
          intervals_[rank].insert(
            std::pair< KeyType, ValueType>(merged, merged));
        }
        else {

          // Otherwise, insert the incoming one.
          intervals_[rank].insert(
            std::pair<KeyType, ValueType>(test_interval, test_interval));
        }
      }
      else {
        // Otherwise, insert the incoming one.
        intervals_[rank].insert(
          std::pair<KeyType, ValueType>(test_interval, test_interval));
      }
      return does_not_exist;
    }
};
}
}

#endif

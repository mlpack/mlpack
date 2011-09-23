/** @file disjoint_int_intervals.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_DISJOINT_INT_INTERVALS_H
#define CORE_PARALLEL_DISJOINT_INT_INTERVALS_H

#include <boost/scoped_array.hpp>
#include <map>

namespace core {
namespace parallel {

class DisjointIntIntervals {

  public:

    typedef std::pair<int, int> ValueType;

    typedef ValueType KeyType;

    struct ComparatorType {
      bool operator()(const KeyType &k1, const KeyType &k2) const {
        return k1.second < k2.first || k2.second < k1.first;
      }
    };

    typedef std::map< KeyType, ValueType, ComparatorType > MapType;

  private:

    boost::scoped_array<MapType> intervals_;

  public:

    long reference_count_;

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

    DisjointIntIntervals() {
      reference_count_ = 0;
    }

    void Init(boost::mpi::communicator &world) {
      boost::scoped_array<MapType> tmp_array(new MapType[ world.size()]);
      intervals_.swap(tmp_array);
    }

    template<typename DistributedTableType>
    bool work_is_complete(
      boost::mpi::communicator &world, DistributedTableType *table_in) {
      bool completed = true;
      for(int i = 0; completed && i < world.size(); i++) {
        std::pair<int, int> key(0, table_in->local_n_entries(i));
        if(intervals_[i].size() == 0) {
          completed = false;
          break;
        }
        typename MapType::iterator it = intervals_[i].find(key);
        completed = (it->first.first == key.first &&
                     it->first.second == key.second);
      }
      return completed;
    }

    DisjointIntIntervals(
      boost::mpi::communicator &world,
      const DisjointIntIntervals &intervals_in) {
      reference_count_ = 0;
      this->Init(world);
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

inline void intrusive_ptr_add_ref(DisjointIntIntervals *ptr) {
  ptr->reference_count_++;
}

inline void intrusive_ptr_release(DisjointIntIntervals *ptr) {
  ptr->reference_count_--;
  if(ptr->reference_count_ == 0) {
    if(core::table::global_m_file_) {
      core::table::global_m_file_->DestroyPtr(ptr);
    }
    else {
      delete ptr;
    }
  }
}
}
}

#endif

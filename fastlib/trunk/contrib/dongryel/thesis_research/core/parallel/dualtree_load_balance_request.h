/** @file dualtree_load_balance_request.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_DUALTREE_LOAD_BALANCE_REQUEST_H
#define CORE_PARALLEL_DUALTREE_LOAD_BALANCE_REQUEST_H

#include <boost/scoped_array.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>

namespace core {
namespace parallel {
template<typename SubTableType>
class DualtreeLoadBalanceRequest {
  private:
    boost::scoped_array< boost::tuple<int, int, int> > query_subtable_ids_;

    unsigned long int remaining_local_computation_;

    int num_existing_query_subtables_;

    unsigned long int remaining_extra_points_to_hold_;

  public:

    unsigned long int remaining_local_computation() const {
      return remaining_local_computation_;
    }

    unsigned long int remaining_extra_points_to_hold() const {
      return remaining_extra_points_to_hold_;
    }

    template<class Archive>
    void save(Archive &ar, const unsigned int version) const {
      ar & num_existing_query_subtables_;
      for(int i = 0; i < num_existing_query_subtables_; i++) {
        ar & query_subtable_ids_[i].get<0>();
        ar & query_subtable_ids_[i].get<1>();
        ar & query_subtable_ids_[i].get<2>();
      }
      ar & remaining_local_computation_;
      ar & remaining_extra_points_to_hold_;
    }

    /** @brief Serialize the object.
     */
    template<class Archive>
    void load(Archive &ar, const unsigned int version) {
      ar & num_existing_query_subtables_;
      if(num_existing_query_subtables_ > 0) {
        boost::scoped_array <
        boost::tuple<int, int, int> > tmp_array(
          new boost::tuple<int, int, int>[
            num_existing_query_subtables_]);
        query_subtable_ids_.swap(tmp_array);
        for(int i = 0; i < num_existing_query_subtables_; i++) {
          ar & query_subtable_ids_[i].get<0>();
          ar & query_subtable_ids_[i].get<1>();
          ar & query_subtable_ids_[i].get<2>();
        }
      }
      ar & remaining_local_computation_;
      ar & remaining_extra_points_to_hold_;
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    DualtreeLoadBalanceRequest() {
      remaining_local_computation_ = 0;
      num_existing_query_subtables_ = 0;
      remaining_extra_points_to_hold_ = 0;
    }

    void Init(
      std::vector< boost::shared_ptr<SubTableType> > &query_subtables,
      unsigned long int remaining_local_computation_in,
      unsigned long int remaining_extra_points_to_hold_in) {

      // Allocate the array.
      num_existing_query_subtables_ = query_subtables.size();
      if(num_existing_query_subtables_ > 0) {
        boost::scoped_array <
        boost::tuple<int, int, int> > tmp_array(
          new boost::tuple<int, int, int>[ num_existing_query_subtables_]);
        query_subtable_ids_.swap(tmp_array);

        for(int i = 0; i < num_existing_query_subtables_ ; i++) {
          int rank = query_subtables[i]->table()->rank();
          int begin = query_subtables[i]->start_node()->begin();
          int count = query_subtables[i]->start_node()->count();
          query_subtable_ids_[i] =
            boost::tuple<int, int, int>(rank, begin, count);
        }

        // Set the current estimated remaining local work.
        remaining_local_computation_ = remaining_local_computation_in;

        // Set the maximum number of points to receive.
        remaining_extra_points_to_hold_ = remaining_extra_points_to_hold_in;
      }
    }

    DualtreeLoadBalanceRequest(
      bool needs_load_balancing_in,
      std::vector< boost::shared_ptr<SubTableType> > &query_subtables,
      unsigned long int remaining_local_computation_in,
      unsigned long int remaining_extra_points_to_hold_in) {

      this->Init(
        needs_load_balancing_in, query_subtables,
        remaining_local_computation_in,
        remaining_extra_points_to_hold_in);
    }
};
}
}

#endif

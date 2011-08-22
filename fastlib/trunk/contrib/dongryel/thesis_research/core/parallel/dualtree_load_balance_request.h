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

    unsigned long int current_remaining_local_computation_;

    bool needs_load_balancing_;

    int num_existing_query_subtables_;

    unsigned long int remaining_extra_points_to_hold_;

  public:

    template<class Archive>
    void save(Archive &ar, const unsigned int version) const {

      // Save the flag.
      ar & needs_load_balancing_;
      if(needs_load_balancing_) {
        ar & num_existing_query_subtables_;
        for(int i = 0; i < num_existing_query_subtables_; i++) {
          ar & query_subtable_ids_[i].get<0>();
          ar & query_subtable_ids_[i].get<1>();
          ar & query_subtable_ids_[i].get<2>();
        }
        ar & current_remaining_local_computation_;
        ar & remaining_extra_points_to_hold_;
      }
    }

    /** @brief Serialize the object.
     */
    template<class Archive>
    void load(Archive &ar, const unsigned int version) {

      // Load the flag.
      ar & needs_load_balancing_;

      if(needs_load_balancing_) {
        ar & num_existing_query_subtables_;
        boost::scoped_array <
        boost::tuple<int, int, int> > tmp_array(
          new boost::tuple<int, int, int>[
            num_existing_query_subtables_]);
        for(int i = 0; i < num_existing_query_subtables_; i++) {
          ar & query_subtable_ids_[i].get<0>();
          ar & query_subtable_ids_[i].get<1>();
          ar & query_subtable_ids_[i].get<2>();
        }
        ar & current_remaining_local_computation_;
        ar & remaining_extra_points_to_hold_;
      }
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    DualtreeLoadBalanceRequest() {
      current_remaining_local_computation_ = 0;
      needs_load_balancing_ = false;
      num_existing_query_subtables_ = 0;
      remaining_extra_points_to_hold_ = 0;
    }

    DualtreeLoadBalanceRequest(
      bool needs_load_balancing_in,
      std::vector< boost::shared_ptr<SubTableType> > &query_subtables,
      unsigned long int current_remaining_local_computation_in,
      unsigned long int remaining_extra_points_to_hold_in) {

      // Set the flag.
      needs_load_balancing_ = needs_load_balancing_in;

      if(needs_load_balancing_) {

        // Allocate the array.
        num_existing_query_subtables_ = query_subtables.size();
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
        current_remaining_local_computation_ =
          current_remaining_local_computation_in;

        // Set the maximum number of points to receive.
        remaining_extra_points_to_hold_ = remaining_extra_points_to_hold_in;
      }
    }
};
}
}

#endif

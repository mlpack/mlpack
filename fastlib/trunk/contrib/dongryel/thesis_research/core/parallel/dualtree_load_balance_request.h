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

    unsigned long int remaining_local_computation_;

    unsigned long int remaining_extra_points_to_hold_;

  public:

    unsigned long int remaining_local_computation() const {
      return remaining_local_computation_;
    }

    unsigned long int remaining_extra_points_to_hold() const {
      return remaining_extra_points_to_hold_;
    }

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & remaining_local_computation_;
      ar & remaining_extra_points_to_hold_;
    }

    DualtreeLoadBalanceRequest() {
      remaining_local_computation_ = 0;
      remaining_extra_points_to_hold_ = 0;
    }

    void Init(
      unsigned long int remaining_local_computation_in,
      unsigned long int remaining_extra_points_to_hold_in) {

      // Set the current estimated remaining local work.
      remaining_local_computation_ = remaining_local_computation_in;

      // Set the maximum number of points to receive.
      remaining_extra_points_to_hold_ = remaining_extra_points_to_hold_in;
    }
};
}
}

#endif

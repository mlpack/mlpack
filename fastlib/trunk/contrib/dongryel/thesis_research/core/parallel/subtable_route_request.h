/** @file subtable_route_request.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_SUBTABLE_ROUTE_REQUEST_H
#define CORE_PARALLEL_SUBTABLE_ROUTE_REQUEST_H

#include <boost/serialization/serialization.hpp>
#include <vector>
#include "core/table/sub_table.h"

namespace core {
namespace parallel {

template<typename TableType>
class SubTableRouteRequest {
  private:
    std::vector<int> destinations_;

    int upper_limit_stage_;

    int stage_;

    core::parallel::SubTable<TableType> sub_table_;

  public:

    int stage() const {
      return stage_;
    }

    int upper_limit_stage() const {
      return upper_limit_stage_;
    }

    void add_destination(int new_dest_in) {
      destinations_.push_back(new_dest_in);
    }

    void set_stage(int stage_in) {
      stage_ = stage_in;
    }

    void set_upper_limit_stage(int upper_limit_stage_in) {
      upper_limit_stage_ = upper_limit_stage_in;
    }

    template<class Archive>
    void save(Archive &ar, const unsigned int version) const {
      ar & stage_;
      int num_destinations = destinations_.size();
      ar & num_destinations;
      for(int i = 0; i < num_destinations; i++) {
        ar & destinations_[i];
      }

      // Save the subtable.
      ar & sub_table_;
    }

    template<class Archive>
    void load(Archive &ar, const unsigned int version) {
      int num_destinations;
      ar & stage_;
      ar & num_destinations;
      destinations_.resize(num_destinations);
      for(int i = 0; i < num_destinations; i++) {
        ar & destinations_[i];
      }

      // Load the subtable.
      ar & sub_table_;
    }

    void GenerateNextRouteRequest(
      core::parallel::SubTableRouteRequest *to_be_routed,
      int *route_destination) {

      // Compute the next routing destination.
      *route_destination = -1;

      // If there is only one destination, then route directly.
      if(destinations_.size() == 1) {
        to_be_routed->set_stage_(upper_limit_stage_);
        *route_destination = destinations[0];
      }
      else {

        // Compute the first differing least significant bit between the
        // source and each destination.
        int first_differing_least_sig_bit_pos =
          std::numeric_limits<int>::max();
        for(unsigned int i = 0; i < destinations.size(); i++) {
          int differing_least_sig_bit_pos =
            core::math::LeastSignificantDifferingBit<int>(
              source, destinations[i], starting_bit_pos,
              upper_limit_stage_);
          if(differing_least_sig_bit_pos < 0) {
            keep_list->push_back(destinations[i]);
          }
          else {
            first_differing_least_sig_bit_pos =
              std::min(
                first_differing_least_sig_bit_pos,
                differing_least_sig_bit_pos);
          }
        }
        to_be_routed->set_stage(first_differing_leaest_sig_bit_pos);
        *route_destination = source ^(1 << (* chosen_bit_pos));
      }
    }

    void Init(
      const std::vector<int> &destinations_in,
      int upper_limit_stage_in,
      int stage_in) {

      destinations_ = destinations_in;
      upper_limit_stage_ = upper_limit_stage_in;
      stage_ = stage_in;
    }
};
}
}

#endif

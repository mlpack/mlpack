/** @file distributed_dualtree_subtable_routing.h
 *
 *  Subtable routing based on the e-cube routing algorithm.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_DISTRIBUTED_DUALTREE_SUBTABLE_ROUTING_H
#define CORE_PARALLEL_DISTRIBUTED_DUALTREE_SUBTABLE_ROUTING_H

#include <vector>
#include "core/math/math_lib.h"

namespace core {
namespace parallel {

class DistributedDualtreeSubtableRouting {

  public:

    static int RouteTo(
      int source, const std::vector<int> &destinations,
      int starting_bit_pos, int upper_limit_bit_pos, int *chosen_bit_pos) {

      int route_destination = -1;

      // If there is only one destination, then route directly.
      if(destinations.size() == 1) {
        *chosen_bit_pos = upper_limit_bit_pos;
        route_destination = destinations[0];
      }
      else {

        // Compute the first differing least significant bit between the
        // source and each destination.
        int first_differing_least_sig_bit_pos = std::numeric_limits<int>::max();
        for(unsigned int i = 0; i < destinations.size(); i++) {
          first_differing_least_sig_bit_pos =
            std::min(
              first_differing_least_sig_bit_pos,
              core::math::LeastSignificantDifferingBit<int>(
                source, destinations[i], starting_bit_pos,
                upper_limit_bit_pos));
        }
        *chosen_bit_pos = first_differing_leaest_sig_bit_pos;
        route_destination = source ^(1 << (* chosen_bit_pos));
      }

      return route_destination;
    }
};
}
}

#endif

/** @file subtable_route_request.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_SUBTABLE_ROUTE_REQUEST_H
#define CORE_PARALLEL_SUBTABLE_ROUTE_REQUEST_H

#include <boost/mpi/communicator.hpp>
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

    core::table::SubTable<TableType> sub_table_;

  public:

    SubTableRouteRequest() {
      upper_limit_stage_ = 0;
      stage_ = 0;
    }

    bool has_same_subtable_id(const std::pair<int, int> &rnode_id) const {

      return sub_table_.start_node()->begin() == rnode_id.first &&
             sub_table_.start_node()->count() == rnode_id.second;
    }

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
      int num_destinations = destinations_.size();
      ar & upper_limit_stage_;
      ar & stage_;
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
      ar & upper_limit_stage_;
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
      boost::mpi::communicator &comm,
      core::parallel::SubTableRouteRequest<TableType> *to_be_routed,
      int *route_destination) {

      // Compute the next routing destination.
      *route_destination = -1;

      // If the current process potentially needs to send a message,
      if(stage_ < upper_limit_stage_) {

        // If there is only one destination, then route directly.
        if(destinations_.size() == 1) {
          to_be_routed->set_stage(upper_limit_stage_);
          to_be_routed->set_upper_limit_stage(upper_limit_stage_);
          *route_destination = destinations_[0];
        }
        else {

          // Compute the first differing least significant bit between the
          // source and each destination.
          int first_differing_least_sig_bit_pos =
            std::numeric_limits<int>::max();
          for(unsigned int i = 0; i < destinations_.size(); i++) {
            int differing_least_sig_bit_pos =
              core::math::LeastSignificantDifferingBit<int>(
                comm.rank(), destinations_[i], stage_, upper_limit_stage_);

            // If any of the bits is different between the range, then
            // add to the list of destinations to be routed.
            if(differing_least_sig_bit_pos >= 0) {
              to_be_routed->add_destination(destinations_[i]);
              first_differing_least_sig_bit_pos =
                std::min(
                  first_differing_least_sig_bit_pos,
                  differing_least_sig_bit_pos);
            }
          }

          // Set the starting stage of the message to be routed.
          to_be_routed->set_stage(first_differing_least_sig_bit_pos);

          // Move to the next stage.
          stage_++;
        }
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

    void InitSubTableForSending(
      TableType *table_in, const std::pair<int, int> &rnode_subtree_id) {

      sub_table_.Init(
        table_in,
        table_in->get_tree()->FindByBeginCount(
          rnode_subtree_id.first,
          rnode_subtree_id.second),
        false);
    }
};
}
}

#endif

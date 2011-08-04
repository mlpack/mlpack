/** @file subtable_route_request.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_SUB_TABLE_ROUTE_REQUEST_H
#define CORE_PARALLEL_SUB_TABLE_ROUTE_REQUEST_H

#include <boost/mpi/communicator.hpp>
#include <boost/serialization/serialization.hpp>
#include <vector>
#include "core/table/sub_table.h"

namespace core {
namespace parallel {

template<typename TableType>
class SubTableRouteRequest {
  public:

    typedef core::parallel::SubTableRouteRequest<TableType> SubTableRouteRequestType;

  private:

    /** @brief The list of MPI ranks for which the subtable should be
     *         forwarded to.
     */
    std::vector<int> destinations_;

    /** @brief The subtable to be routed.
     */
    core::table::SubTable<TableType> sub_table_;

    /** @brief An internal variable to keep track of the number of
     *         destinations passed to the next routing destination.
     */
    int num_routed_;

    /** @brief The next destination to which the next routing message
     *         should be forwarded.
     */
    int next_destination_;

    /** @brief The MPI rank of the calling process.
     */
    int rank_;

    /** @brief The beginning index and the count of the destinations,
     *         roughly corresponds to one of the nodes in the MPI
     *         process binary tree.
     */
    std::pair<int, int> begin_count_pair_;

  public:

    const std::pair<int, int> &begin_count_pair() const {
      return begin_count_pair_;
    }

    int rank() const {
      return rank_;
    }

    int next_destination() const {
      return next_destination_;
    }

    /** @brief Returns the number of destinations passed to the next
     *         routing destination after an asynchronous send is
     *         issued.
     */
    int num_routed() const {
      return num_routed_;
    }

    int num_destinations() const {
      return destinations_.size();
    }

    bool remove_from_destination_list(int destination_in) {
      typename std::vector<int>::iterator it =
        std::find(
          destinations_.begin(), destinations_.end(), destination_in);
      if(it != destinations_.end()) {
        (*it) = destinations_.back();
        destinations_.pop_back();
        return true;
      }
      else {
        return false;
      }
    }

    SubTableRouteRequest() {
      num_routed_ = 0;
      next_destination_ = 0;
      rank_ = 0;
      begin_count_pair_ = std::pair<int, int>(0, 0);
    }

    core::table::SubTable<TableType> &sub_table() {
      return sub_table_;
    }

    const core::table::SubTable<TableType> &sub_table() const {
      return sub_table_;
    }

    const std::vector<int> &destinations() const {
      return destinations_;
    }

    bool has_same_subtable_id(const std::pair<int, int> &rnode_id) const {
      return sub_table_.start_node()->begin() == rnode_id.first &&
             sub_table_.start_node()->count() == rnode_id.second;
    }

    void add_destination(int new_dest_in) {
      destinations_.push_back(new_dest_in);
    }

    template<class Archive>
    void save(Archive &ar, const unsigned int version) const {

      core::parallel::SubTableRouteRequest<TableType> *this_modifiable =
        const_cast< core::parallel::SubTableRouteRequest<TableType> * >(this);

      // Push everything in the list.
      this_modifiable->num_routed_ = 0;
      int size = destinations_.size();
      ar & size;
      for(unsigned int i = 0; i < destinations_.size(); i++) {
        ar & destinations_[i];
        this_modifiable->num_routed_++;
      }
      this_modifiable->destinations_.resize(0);

      // Save the subtable.
      ar & sub_table_;
    }

    template<class Archive>
    void load(Archive &ar, const unsigned int version) {

      // Load the size.
      int size;
      ar & size;
      destinations_.resize(size);
      for(int i = 0; i < size; i++) {
        ar & destinations_[i];
      }

      // Load the subtable.
      ar & sub_table_;
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    void Init(
      const SubTableRouteRequestType &source_in) {
      sub_table_.Alias(source_in.sub_table());
      destinations_ = source_in.destinations();
      next_destination_ = source_in.next_destination();
      rank_ = source_in.rank();
    }

    void InitSubTableForReceiving(int cache_id) {
      sub_table_.Init(cache_id, false);
    }

    void InitSubTableForSending(
      boost::mpi::communicator &comm,
      TableType *table_in, const std::pair<int, int> &rnode_subtree_id) {

      sub_table_.Init(
        table_in,
        table_in->get_tree()->FindByBeginCount(
          rnode_subtree_id.first,
          rnode_subtree_id.second),
        false);
    }

    int ComputeNextDestination(boost::mpi::communicator &comm) {
      rank_ = comm.rank();
      int offset = static_cast<int>(ceil(begin_count_pair_.second * 0.5));
      int threshold = begin_count_pair_.first + offset;
      int upper_limit = begin_count_pair_.first +
                        begin_count_pair_.second - 1;

      // Determine which half the process falls into and the next
      // destination based on it.
      if(rank_ <= threshold) {
        next_destination_ = std::min(rank_ + offset, upper_limit);
      }
      else {
        next_destination_ = rank_ - offset;
      }
      return next_destination_;
    }
};
}
}

#endif

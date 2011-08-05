/** @file route_request.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_ROUTE_REQUEST_H
#define CORE_PARALLEL_ROUTE_REQUEST_H

#include <boost/mpi/communicator.hpp>
#include <boost/serialization/serialization.hpp>
#include <vector>
#include "core/table/sub_table.h"

namespace core {
namespace parallel {

template<typename ObjectType>
class RouteRequest {
  public:

    typedef core::parallel::RouteRequest<ObjectType> RouteRequestType;

  private:

    /** @brief The beginning index and the count of the destinations,
     *         roughly corresponds to one of the nodes in the MPI
     *         process binary tree.
     */
    std::pair<int, int> begin_count_pair_;

    /** @brief The list of MPI ranks for which the subtable should be
     *         forwarded to.
     */
    std::vector<int> destinations_;

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

    /** @brief The object to be routed.
     */
    ObjectType object_;

    /** @brief The threshold used for determining whether the calling
     *         process belongs to the left or the right. If it is
     *         above this threshold, then it belongs to the
     *         right. Otherwise to the left.
     */
    int threshold_;

  private:

    void ComputeNextDestination_() {
      int offset = static_cast<int>(ceil(begin_count_pair_.second * 0.5));
      threshold_ = begin_count_pair_.first + offset - 1;
      int upper_limit = begin_count_pair_.first +
                        begin_count_pair_.second - 1;

      // Determine which half the process falls into and the next
      // destination based on it.
      if(rank_ <= threshold_) {
        next_destination_ = std::min(rank_ + offset, upper_limit);
      }
      else {
        next_destination_ = rank_ - offset;
      }
    }

  public:

    const std::pair<int, int> &begin_count_pair() const {
      return begin_count_pair_;
    }

    int rank() const {
      return rank_;
    }

    int next_destination(boost::mpi::communicator &comm) const {
      RouteRequestType *this_modifiable =
        const_cast < RouteRequestType * >(this);
      this_modifiable->rank_ = comm.rank();
      this_modifiable->ComputeNextDestination_();
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

    RouteRequest() {
      num_routed_ = 0;
      next_destination_ = 0;
      rank_ = 0;
      begin_count_pair_ = std::pair<int, int>(0, 0);
      threshold_ = 0;
    }

    ObjectType &object() {
      return object_;
    }

    const ObjectType &object() const {
      return object_;
    }

    const std::vector<int> &destinations() const {
      return destinations_;
    }

    void add_destination(int new_dest_in) {
      destinations_.push_back(new_dest_in);
    }

    template<class Archive>
    void save(Archive &ar, const unsigned int version) const {
      RouteRequestType *this_modifiable =
        const_cast< RouteRequestType * >(this);

      // Count how many messages were routed.
      std::vector<int> filtered;
      if(rank_ <= threshold_) {
        int right_begin = threshold_ + 1;
        int right_count = begin_count_pair_.first +
                          begin_count_pair_.second - (threshold_ + 1) ;
        ar & right_begin;
        ar & right_count;
        this_modifiable->begin_count_pair_ =
          std::pair<int, int>(
            begin_count_pair_.first, begin_count_pair_.second - right_count);

        // Route to the destination which are larger than the threshold.
        for(int i = 0; i < static_cast<int>(destinations_.size()); i++) {
          if(destinations_[i] > threshold_) {
            filtered.push_back(destinations_[i]);
            this_modifiable->destinations_[i] = destinations_.back();
            this_modifiable->destinations_.pop_back();
            i--;
          }
        }
      }
      else {

        int left_count = threshold_ - begin_count_pair_.first + 1;
        ar & begin_count_pair_.first;
        ar & left_count;
        this_modifiable->begin_count_pair_ =
          std::pair<int, int>(
            threshold_ + 1, begin_count_pair_.second - left_count);

        // Route to the destination which are at most the threshold.
        for(int i = 0; i < static_cast<int>(destinations_.size()); i++) {
          if(destinations_[i] <= threshold_) {
            filtered.push_back(destinations_[i]);
            this_modifiable->destinations_[i] = destinations_.back();
            this_modifiable->destinations_.pop_back();
            i--;
          }
        }
      }
      this_modifiable->num_routed_ = filtered.size();
      ar & num_routed_;
      for(int i = 0; i < num_routed_; i++) {
        ar & filtered[i];
      }

      // Save the object, only if the number of routed messages is
      // at least 1.
      if(filtered.size() > 0) {
        ar & object_;
      }
    }

    template<class Archive>
    void load(Archive &ar, const unsigned int version) {

      // Load the begin and count pair that includes the receiving MPI
      // rank.
      ar & begin_count_pair_.first;
      ar & begin_count_pair_.second;

      // Load the size.
      int size;
      ar & size;
      destinations_.resize(size);
      for(int i = 0; i < size; i++) {
        ar & destinations_[i];
      }

      // Load the object, if the message is not empty.
      if(size > 0) {
        ar & object_;
      }
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    void Init(
      boost::mpi::communicator &comm,
      const RouteRequestType &source_in) {
      begin_count_pair_ = source_in.begin_count_pair();
      destinations_ = source_in.destinations();
      next_destination_ = source_in.next_destination(comm);
      num_routed_ = 0;
      rank_ = comm.rank();
      object_ = source_in.object();
    }

    void InitSubTableForReceiving(
      boost::mpi::communicator &comm, int cache_id) {
      rank_ = comm.rank();
      object_.Init(cache_id, false);
    }

    template<typename TableType>
    void InitSubTableForSending(
      boost::mpi::communicator &comm,
      TableType *table_in, const std::pair<int, int> &rnode_subtree_id) {

      begin_count_pair_ = std::pair<int, int>(0, comm.size());
      rank_ = comm.rank();
      object_.Init(
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

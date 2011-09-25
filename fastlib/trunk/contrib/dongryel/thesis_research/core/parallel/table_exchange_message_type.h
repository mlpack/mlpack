/** @file table_exchange_message_type.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_TABLE_EXCHANGE_MESSAGE_TYPE_H
#define CORE_PARALLEL_TABLE_EXCHANGE_MESSAGE_TYPE_H

#include <boost/serialization/serialization.hpp>

namespace core {
namespace parallel {

/** @brief The message type used in the recursive doubling
 *         exchange.
 */
template <
typename EnergyRouteRequestType,
         typename ExtraTaskRouteRequestType,
         typename LoadBalanceRouteRequestType,
         typename SubTableRouteRequestType
         >
class MessageType {
  private:

    // For serialization.
    friend class boost::serialization::access;

  private:

    EnergyRouteRequestType energy_route_;

    ExtraTaskRouteRequestType extra_task_route_;

    SubTableRouteRequestType flush_route_;

    LoadBalanceRouteRequestType load_balance_route_;

    int originating_rank_;

    SubTableRouteRequestType subtable_route_;

  public:

    MessageType() {
      originating_rank_ = 0;
    }

    void operator=(const MessageType &message_in) {
      energy_route_ = message_in.energy_route();
      extra_task_route_ = message_in.extra_task_route();
      flush_route_ = message_in.flush_route();
      load_balance_route_ = message_in.load_balance_route();
      originating_rank_ = message_in.originating_rank();
      subtable_route_ = message_in.subtable_route();
    }

    MessageType(const MessageType &message_in) {
      this->operator=(message_in);
    }

    int next_destination(boost::mpi::communicator &comm) {
      energy_route_.next_destination(comm);
      extra_task_route_.next_destination(comm);
      flush_route_.next_destination(comm);
      load_balance_route_.next_destination(comm);
      return subtable_route_.next_destination(comm);
    }

    void set_originating_rank(int rank_in) {
      originating_rank_ = rank_in;
    }

    int originating_rank() const {
      return originating_rank_;
    }

    ExtraTaskRouteRequestType &extra_task_route() {
      return extra_task_route_;
    }

    const ExtraTaskRouteRequestType &extra_task_route() const {
      return extra_task_route_;
    }

    LoadBalanceRouteRequestType &load_balance_route() {
      return load_balance_route_;
    }

    const LoadBalanceRouteRequestType &load_balance_route() const {
      return load_balance_route_;
    }

    SubTableRouteRequestType &subtable_route() {
      return subtable_route_;
    }

    const SubTableRouteRequestType &subtable_route() const {
      return subtable_route_;
    }

    EnergyRouteRequestType &energy_route() {
      return energy_route_;
    }

    const EnergyRouteRequestType &energy_route() const {
      return energy_route_;
    }

    SubTableRouteRequestType &flush_route() {
      return flush_route_;
    }

    const SubTableRouteRequestType &flush_route() const {
      return flush_route_;
    }

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & energy_route_;
      ar & extra_task_route_;
      ar & flush_route_;
      ar & load_balance_route_;
      ar & originating_rank_;
      ar & subtable_route_;
    }
};
}
}

#endif

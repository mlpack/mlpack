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
template < typename SubTableRouteRequestType,
         typename EnergyRouteRequestType >
class MessageType {
  private:

    // For serialization.
    friend class boost::serialization::access;

  private:
    int originating_rank_;

    SubTableRouteRequestType subtable_route_;

    EnergyRouteRequestType energy_route_;

    SubTableRouteRequestType flush_route_;

    bool serialize_subtable_route_;

  public:

    bool serialize_subtable_route() const {
      return serialize_subtable_route_;
    }

    void set_serialize_subtable_route_flag(bool flag_in) {
      serialize_subtable_route_ = flag_in;
    }

    MessageType() {
      originating_rank_ = 0;
      serialize_subtable_route_ = true;
    }

    void operator=(const MessageType &message_in) {
      originating_rank_ = message_in.originating_rank();
      subtable_route_ = message_in.subtable_route();
      energy_route_ = message_in.energy_route();
      flush_route_ = message_in.flush_route();
      serialize_subtable_route_ = message_in.serialize_subtable_route_;
    }

    void CopyWithoutSubTableRoute(const MessageType &message_in) {
      originating_rank_ = message_in.originating_rank();
      energy_route_ = message_in.energy_route();
      flush_route_ = message_in.flush_route();
      serialize_subtable_route_ = false;
    }

    MessageType(const MessageType &message_in) {
      this->operator=(message_in);
    }

    int next_destination(boost::mpi::communicator &comm) {
      subtable_route_.next_destination(comm);
      energy_route_.next_destination(comm);
      return flush_route_.next_destination(comm);
    }

    void set_originating_rank(int rank_in) {
      originating_rank_ = rank_in;
    }

    int originating_rank() const {
      return originating_rank_;
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
    void save(Archive &ar, const unsigned int version) const {
      ar & originating_rank_;
      ar & serialize_subtable_route_;
      if(serialize_subtable_route_) {
        ar & subtable_route_;
      }
      ar & energy_route_;
      ar & flush_route_;
    }

    template<class Archive>
    void load(Archive &ar, const unsigned int version) {
      ar & originating_rank_;
      ar & serialize_subtable_route_;
      if(serialize_subtable_route_) {
        ar & subtable_route_;
      }
      ar & energy_route_;
      ar & flush_route_;
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};
}
}

#endif

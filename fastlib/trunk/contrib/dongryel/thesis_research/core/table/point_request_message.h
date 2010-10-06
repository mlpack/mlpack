/** @file point_request_message.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_POINT_REQUEST_MESSAGE_H
#define CORE_TABLE_POINT_REQUEST_MESSAGE_H

namespace core {
namespace table {

class PointRequestMessage {
  private:

    int source_rank_;

    int point_id_;

    friend class boost::serialization::access;

  public:

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & source_rank_;
      ar & point_id_;
    }

    PointRequestMessage() {
      Reset();
    }

    PointRequestMessage(int source_rank_in, int point_id_in) {
      source_rank_ = source_rank_in;
      point_id_ = point_id_in;
    }

    void Reset() {
      source_rank_ = -1;
      point_id_ = -1;
    }

    int source_rank() const {
      return source_rank_;
    }

    int point_id() const {
      return point_id_;
    }
};
};
};

#endif

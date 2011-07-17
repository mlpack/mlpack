/** @file subtable_send_request.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_SUBTABLE_SEND_REQUEST_H
#define CORE_PARALLEL_SUBTABLE_SEND_REQUEST_H

namespace core {
namespace parallel {

class SubTableSendRequest {
  private:
    int destination_;

    int begin_;

    int count_;

    double priority_;

  public:

    void operator=(const SubTableSendRequest &request_in) {
      destination_ = request_in.destination();
      begin_ = request_in.begin();
      count_ = request_in.count();
      priority_ = request_in.priority();
    }

    SubTableSendRequest(const SubTableSendRequest &request_in) {
      this->operator=(request_in);
    }

    SubTableSendRequest() {
      destination_ = 0;
      begin_ = 0;
      count_ = 0;
      priority_ = 0.0;
    }

    SubTableSendRequest(
      int destination_in, int begin_in, int count_in, double priority_in) {

      destination_ = destination_in;
      begin_ = begin_in;
      count_ = count_in;
      priority_ = priority_in;
    }

    int destination() const {
      return destination_;
    }

    int begin() const {
      return begin_;
    }

    int count() const {
      return count_;
    }

    double priority() const {
      return priority_;
    }
};
}
}

#endif

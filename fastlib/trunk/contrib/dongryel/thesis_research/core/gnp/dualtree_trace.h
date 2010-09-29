/** @file dualtree_trace.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_GNP_DUALTREE_TRACE_H
#define CORE_GNP_DUALTREE_TRACE_H

#include <deque>

namespace ml {

template<typename ArgType>
class DualtreeTrace {

  private:

    std::deque<ArgType> trace_;

  public:

    void push_front(const ArgType &arg_in) {
      trace_.push_front(arg_in);
    }

    void pop_front() {
      trace_.pop_front();
    }

    void push_back(const ArgType &arg_in) {
      trace_.push_back(arg_in);
    }

    void pop_back() {
      trace_.pop_back();
    }

    ArgType &back() {
      return trace_.back();
    }

    const ArgType &back() const {
      return trace_.back();
    }

    ArgType &front() {
      return trace_.front();
    }

    const ArgType &front() const {
      return trace_.front();
    }

    bool empty() const {
      return trace_.empty();
    }

    /** @brief Initialize the dual-tree trace.
     */
    void Init() {
      trace_.resize(0);
    }

};
};

#endif

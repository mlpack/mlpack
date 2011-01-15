/** @file dualtree_trace.h
 *
 *  An auxilary class for doing an iterative version of dualtree
 *  computations.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_GNP_DUALTREE_TRACE_H
#define CORE_GNP_DUALTREE_TRACE_H

#include <deque>

namespace core {
namespace gnp {

template<typename ArgType>
class DualtreeTrace {

  private:

    /** @brief The trace object for maintaining a stack of dual-tree
     *         computations.
     */
    std::deque<ArgType> trace_;

  public:

    /** @brief Pushes a computation object in the front.
     */
    void push_front(const ArgType &arg_in) {
      trace_.push_front(arg_in);
    }

    /** @brief Pops the front computation object.
     */
    void pop_front() {
      trace_.pop_front();
    }

    /** @brief Pushes a computation object at the back.
     */
    void push_back(const ArgType &arg_in) {
      trace_.push_back(arg_in);
    }

    /** @brief Pops a computation object at the back.
     */
    void pop_back() {
      trace_.pop_back();
    }

    /** @brief Returns the back of the trace.
     */
    ArgType &back() {
      return trace_.back();
    }

    /** @brief Returns the back of the trace.
     */
    const ArgType &back() const {
      return trace_.back();
    }

    /** @brief Returns the front of the trace.
     */
    ArgType &front() {
      return trace_.front();
    }

    /** @brief Returns the front of the trace.
     */
    const ArgType &front() const {
      return trace_.front();
    }

    /** @brief Returns whether the trace is empty or not.
     */
    bool empty() const {
      return trace_.empty();
    }

    /** @brief Initialize the dual-tree trace.
     */
    void Init() {
      trace_.resize(0);
    }
};
}
}

#endif

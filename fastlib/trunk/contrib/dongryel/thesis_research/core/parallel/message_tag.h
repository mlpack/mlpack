/** @file message_tag.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_MESSAGE_TAG_H
#define CORE_PARALLEL_MESSAGE_TAG_H

namespace core {
namespace parallel {

class MessageTag {
  public:
    enum MessageTagType {
      ROUTE_SUBTABLE, LOAD_BALANCE_REQUEST, TASK_LIST, FINISHED_TUPLES
    };
};
}
}

#endif

/** @file message_tag.h
 *
 *  Defines the tags used for MPI calls in distributed code.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_MESSAGE_TAG_H
#define CORE_PARALLEL_MESSAGE_TAG_H

namespace core {
namespace parallel {

/** @brief The list of tags used for MPI calls in distributed code.
 */
class MessageTag {
  public:
    enum MessageTagType { ROUTE_SUBTABLE, FINISHED_TUPLES };
};
}
}

#endif

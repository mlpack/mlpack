/** @file distributed_table_message.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_DISTRIBUTED_TABLE_MESSAGE_H
#define CORE_TABLE_DISTRIBUTED_TABLE_MESSAGE_H

namespace core {
namespace table {
class DistributedTableMessage {
  public:
    enum DistributedTableRequest { REQUEST_POINT, RECEIVE_POINT, TERMINATE_POINT_INBOX, TERMINATE_POINT_REQUEST_MESSAGE_INBOX, TERMINATE_POINT_REQUEST_MESSAGE_OUTBOX };
};
};
};

#endif

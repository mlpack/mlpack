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
    enum DistributedTableRequest { REQUEST_POINT, RECEIVE_POINT, TERMINATE_SERVER };
};
};
};

#endif

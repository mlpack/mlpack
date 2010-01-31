/**
 * @file sched.cc
 *
 * Non-templated implementations for GNP scheduling.
 */

#include "sched.h"

void SchedulerInterface::Report(struct datanode *module) {
}

//--------------------------------------------------------------

void RemoteSchedulerBackend::Init(SchedulerInterface *inner_work_queue) {
  inner_ = inner_work_queue;
}

void RemoteSchedulerBackend::HandleRequest(
    const WorkRequest& request, WorkResponse *response) {
  DEBUG_ASSERT(request.operation == WorkRequest::GIVE_ME_WORK);
  inner_->GetWork(request.rank, &response->work_items);
}

//--------------------------------------------------------------

void RemoteScheduler::Init(int channel, int destination) {
  channel_ = channel;
  destination_ = destination;
}

void RemoteScheduler::GetWork(
    int rank, ArrayList<Grain> *work_items) {
  WorkRequest request;
  request.operation = WorkRequest::GIVE_ME_WORK;
  request.rank = rank;
  Rpc<WorkResponse> response(channel_, destination_, request);
  work_items->InitCopy(response->work_items);
}

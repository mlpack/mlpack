#include "work.h"

void WorkQueueInterface::Report(struct datanode *module) {
}

//--------------------------------------------------------------

void RemoteWorkQueueBackend::Init(WorkQueueInterface *inner_work_queue) {
  inner_ = inner_work_queue;
}

void RemoteWorkQueueBackend::HandleRequest(
    const WorkRequest& request, WorkResponse *response) {
  DEBUG_ASSERT(request.operation == WorkRequest::GIVE_ME_WORK);
  inner_->GetWork(request.rank, &response->work_items);
}

//--------------------------------------------------------------

void RemoteWorkQueue::Init(int channel, int destination) {
  channel_ = channel;
  destination_ = destination;
}

void RemoteWorkQueue::GetWork(
    int rank, ArrayList<Grain> *work_items) {
  WorkRequest request;
  request.operation = WorkRequest::GIVE_ME_WORK;
  request.rank = rank;
  Rpc<WorkResponse> response(channel_, destination_, request);
  work_items->Copy(response->work_items);
}

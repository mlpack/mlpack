#include "work.h"

#ifdef USE_MPI
void RemoteWorkQueueBackend::Init(WorkQueueInterface *inner_work_queue) {
  inner_ = inner_work_queue;
}

void RemoteWorkQueueBackend::HandleRequest(
    const WorkRequest& request, WorkResponse *response) {
  DEBUG_ASSERT(request.operation == WorkRequest::GIVE_ME_WORK);
  inner_->GetWork(&response->work_items);
}


void RemoteWorkQueue::Init(int channel, int destination) {
  channel_ = channel;
  destination_ = destination;
}

void RemoteWorkQueue::GetWork(ArrayList<index_t> *work_items) {
  WorkRequest request;
  request.operation = WorkRequest::GIVE_ME_WORK;
  Rpc<WorkResponse> response(channel_, destination_, request);
  work_items->Copy(response->work_items);
}
#endif

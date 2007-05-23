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
  stub_.Init(channel, destination);
}

void RemoteWorkQueue::GetWork(ArrayList<index_t> *work_items) {
  WorkRequest request;
  request.operation = WorkRequest::GIVE_ME_WORK;

  stub_.Lock();
  const WorkResponse *response = stub_.Request(request);
  ot::Print(*response);
  work_items->Copy(response->work_items);
  stub_.Unlock();
}
#endif

#include "rpc.h"

int main(int argc, char *argv[]) {
  fx_init(argc, argv);
  rpc::Init();
  fprintf(stderr, "Initialized.\n");
  
  if (rpc::rank() == 0) {
    ArrayList<int> mylist;
    mylist.Init();
    *mylist.AddBack() = 5;
    DataGetterBackend<ArrayList<int> > backend;
    backend.Init(&mylist);
    rpc::Register(10, &backend);
    rpc::Done();
  } else {
    ArrayList<int> mylist;
    rpc::GetRemoteData(10, 0, &mylist);
    DEBUG_ASSERT(mylist[0] == 5);
    rpc::Done();
  }
  
  fprintf(stderr, "Done.\n");
  fx_done();
}


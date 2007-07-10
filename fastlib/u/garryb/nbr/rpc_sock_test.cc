#include "rpc.h"

int main(int argc, char *argv[]) {
  fx_init(argc, argv);
  rpc::Init();
  fprintf(stderr, "Initialized.\n");
  rpc::Done();
  fprintf(stderr, "Done.\n");
  fx_done();
}


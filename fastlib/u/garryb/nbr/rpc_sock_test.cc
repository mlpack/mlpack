#include "rpc.h"

int main(int argc, char *argv[]) {
  fx_init(argc, argv);
  rpc::Init();
  rpc::Done();
  fx_done();
}


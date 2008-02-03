#include "fastlib/fastlib.h"
#include "local_linear_krylov.h"

int main(int argc, char *argv[]) {

  // Initialize FastExec...
  fx_init(argc, argv);

  // Declare local linear krylov object.
  LocalLinearKrylov<GaussianKernel> local_linear;
  
  // Finalize FastExec and print output results.
  fx_done();
  return 0;
}

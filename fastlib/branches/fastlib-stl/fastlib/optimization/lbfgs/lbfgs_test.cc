/***
 * @file lbfgs_test.cc
 *
 * Tests the L-BFGS optimizer on a couple test functions.
 *
 * @author Ryan Curtin (gth671b@mail.gatech.edu)
 */

#include <fastlib/fastlib.h>
#include "lbfgs.h"
#include "test_functions.h"

using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

bool TestRosenbrockFunction() {
  NOTIFY("Testing Rosenbrock function...");

  RosenbrockFunction f;
  L_BFGS<RosenbrockFunction> lbfgs;
  lbfgs.Init(f, 10); // 10 memory points

  NOTIFY("Using 10 memory points...");

  arma::vec coords = f.GetInitialPoint();
  if(!lbfgs.Optimize(50, coords))
    NOTIFY("Optimization reported failure.");

  double final_value = f.Evaluate(coords);

  NOTIFY("Final objective value is %lf at (%lf, %lf)",
    final_value, coords[0], coords[1]);

  if((std::abs(final_value) <= 1e-5) &&
     (std::abs(coords[0] - 1) <= 1e-5) &&
     (std::abs(coords[1] - 1) <= 1e-5))
    return true;
  else
    return false;
}

int main(int argc, char* argv[]) {
  fx_init(argc, argv, NULL);

  if(!TestRosenbrockFunction())
    FATAL("Test failed!");
  else
    NOTIFY("Test passed.");

  fx_done(NULL);
}

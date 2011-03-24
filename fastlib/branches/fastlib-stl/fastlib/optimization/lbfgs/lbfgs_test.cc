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

//
// I do not like this test infrastructure.  We need to design something better
// (or use something that someone else has designed that is better).
//

bool TestRosenbrockFunction() {
  NOTIFY("Testing Rosenbrock function...");

  RosenbrockFunction f;
  L_BFGS<RosenbrockFunction> lbfgs;
  lbfgs.Init(f, 10); // 10 memory points

  arma::vec coords = f.GetInitialPoint();
  if(!lbfgs.Optimize(0, coords))
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

bool TestWoodFunction() {
  NOTIFY("Testing Wood function...");

  WoodFunction f;
  L_BFGS<WoodFunction> lbfgs;
  lbfgs.Init(f, 10);

  arma::vec coords = f.GetInitialPoint();
  if(!lbfgs.Optimize(0, coords))
    NOTIFY("Optimization reported failure.");

  double final_value = f.Evaluate(coords);

  NOTIFY("Final objective value is %lf at (%lf, %lf, %lf, %lf)",
    final_value, coords[0], coords[1], coords[2], coords[3]);

  if((std::abs(final_value) <= 1e-5) &&
     (std::abs(coords[0] - 1) <= 1e-5) &&
     (std::abs(coords[1] - 1) <= 1e-5) &&
     (std::abs(coords[2] - 1) <= 1e-5) &&
     (std::abs(coords[3] - 1) <= 1e-5))
    return true;
  else
    return false;
}

bool TestGeneralizedRosenbrockFunction() {
  for (int i = 2; i < 10; i++) {
    // Dimension: powers of 2
    int dim = std::pow(2, i);

    NOTIFY("Testing GeneralizedRosenbrockFunction (%d dimensions)...", dim);

    GeneralizedRosenbrockFunction f(dim);
    L_BFGS<GeneralizedRosenbrockFunction> lbfgs;
    lbfgs.Init(f, 20); // arbitrary choice of memory

    arma::vec coords = f.GetInitialPoint();
    if(!lbfgs.Optimize(0, coords))
      NOTIFY("Optimization reported failure.");

    double final_value = f.Evaluate(coords);

    NOTIFY("Final objective value is %lf.", final_value);

    bool correct = true;
    for (int j = 0; j < dim; j++) {
      if (std::abs(coords[j] - 1) > 1e-5) {
        correct = false;
        break;
      }
    }
    if (std::abs(final_value) > 1e-5)
      correct = false;

    if (correct)
      NOTIFY("Test passed.");
    else
      FATAL("Test failed!");
  }

  return true;
};

bool TestRosenbrockWoodFunction() {
  NOTIFY("Testing Rosenbrock-Wood combined function...");

  RosenbrockWoodFunction f;
  L_BFGS<RosenbrockWoodFunction> lbfgs;
  lbfgs.Init(f, 10);

  arma::mat coords = f.GetInitialPoint();
  if(!lbfgs.Optimize(0, coords))
    NOTIFY("Optimization reported failure.");

  double final_value = f.Evaluate(coords);

  NOTIFY("Final objective value is %lf at", final_value);
  NOTIFY("  [[%lf, %lf, %lf, %lf]", coords[0, 0], coords[1, 0], coords[2, 0],
      coords[3, 0]);
  NOTIFY("   [%lf, %lf, %lf, %lf]]", coords[0, 1], coords[1, 1], coords[2, 1],
      coords[3, 1]);

  if((std::abs(final_value) <= 1e-5) &&
     (std::abs(coords[0, 0] - 1) <= 1e-5) &&
     (std::abs(coords[1, 0] - 1) <= 1e-5) &&
     (std::abs(coords[2, 0] - 1) <= 1e-5) &&
     (std::abs(coords[3, 0] - 1) <= 1e-5) &&
     (std::abs(coords[0, 1] - 1) <= 1e-5) &&
     (std::abs(coords[1, 1] - 1) <= 1e-5) &&
     (std::abs(coords[2, 1] - 1) <= 1e-5) &&
     (std::abs(coords[3, 1] - 1) <= 1e-5))
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

  if(!TestWoodFunction())
    FATAL("Test failed!");
  else
    NOTIFY("Test passed.");

  TestGeneralizedRosenbrockFunction();

  if(!TestRosenbrockWoodFunction())
    FATAL("Test failed!");
  else
    NOTIFY("Test passed.");

  fx_done(NULL);
}

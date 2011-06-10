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
  IO::Info << "Testing Rosenbrock function... " << std::endl;

  RosenbrockFunction f;
  L_BFGS<RosenbrockFunction> lbfgs(f, 10);

  arma::vec coords = f.GetInitialPoint();
  if(!lbfgs.Optimize(0, coords))
    IO::Info << "Optimization reported failure. " << std::endl;

  double final_value = f.Evaluate(coords);

  IO::Info << "Final objective value is " << final_value << " at(" 
   << coords[0] << ", " << coords[1] << ") " << std::endl;

  if((std::abs(final_value) <= 1e-5) &&
     (std::abs(coords[0] - 1) <= 1e-5) &&
     (std::abs(coords[1] - 1) <= 1e-5))
    return true;
  else
    return false;
}

bool TestWoodFunction() {
  IO::Info << "Testing Wood function... " << std::endl;

  WoodFunction f;
  L_BFGS<WoodFunction> lbfgs(f, 10);

  arma::vec coords = f.GetInitialPoint();
  if(!lbfgs.Optimize(0, coords))
    IO::Info << "Optimization reported failure. " << std::endl;

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

    IO::Info << "Testing GeneralizedRosenbrockFunction (" 
      << dim << " dimensions)... " << std::endl;

    GeneralizedRosenbrockFunction f(dim);
    L_BFGS<GeneralizedRosenbrockFunction> lbfgs(f, 20);

    arma::vec coords = f.GetInitialPoint();
    if(!lbfgs.Optimize(0, coords))
      IO::Info << "Optimization reported failure. " << std::endl;

    double final_value = f.Evaluate(coords);

    IO::Info << "Final objective value is " << final_value << "." << std::endl;

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
      IO::Info << "Test passed. " << std::endl;
    else
      IO::Fatal << "Test failed! " << std::endl;
  }

  return true;
};

bool TestRosenbrockWoodFunction() {
  IO::Info << "Testing Rosenbrock-Wood combined function... " << std::endl;
  RosenbrockWoodFunction f;
  L_BFGS<RosenbrockWoodFunction> lbfgs(f, 10);

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
    IO::Fatal << "Test failed! " << std::endl;
  else
    IO::Info << "Test passed. " << std::endl;

  if(!TestWoodFunction())
    IO::Fatal << "Test failed! " << std::endl;
  else
    IO::Info << "Test passed. " << std::endl;

  TestGeneralizedRosenbrockFunction();

  if(!TestRosenbrockWoodFunction())
    IO::Fatal << "Test failed! " << std::endl;
  else
    IO::Info << "Test passed. " << std::endl;

  fx_done(NULL);
}

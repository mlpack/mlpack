/***
 * @file aug_lagrangian_test.cc
 * @author Ryan Curtin
 *
 * Test of the AugmentedLagrangian class using the test functions defined in
 * aug_lagrangian_test_functions.h.
 */

#include <fastlib/fastlib.h>
#include "aug_lagrangian.h"
#include "aug_lagrangian_test_functions.h"

using namespace mlpack;
using namespace mlpack::optimization;

// 
// I do not like this test infrastructure.  We need to design something better
// (or use something that someone else has designed that is better).
//

bool TestAugLagrangianTestFunction() {
  NOTIFY("Testing AugmentedLagrangianTestFunction...");

  // The choice of 10 memory slots is arbitrary.
  AugLagrangianTestFunction f;
  AugLagrangian<AugLagrangianTestFunction> aug(f, 10);

  arma::vec coords = f.GetInitialPoint();

  if(!aug.Optimize(0, coords))
    NOTIFY("Optimization reported failure.");
  
  double final_value = f.Evaluate(coords);

  NOTIFY("Final objective value is %lf at (%lf, %lf)",
      final_value, coords[0], coords[1]);

  if((std::abs(final_value) <= 1e-5) &&
     (std::abs(coords[0] - 1) <= 1e-5) &&
     (std::abs(coords[1] - 4) <= 1e-5))
    return true;
  else
    return false;
}

int main(int argc, char* argv[]) {
  fx_init(argc, argv, NULL);

  if(!TestAugLagrangianTestFunction())
    FATAL("Test failed!");
  else
    NOTIFY("Test passed.");

  fx_done(NULL);
}

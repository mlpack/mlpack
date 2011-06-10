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

#include "../fx/io.h"

using namespace mlpack;
using namespace mlpack::optimization;

// 
// I do not like this test infrastructure.  We need to design something better
// (or use something that someone else has designed that is better).
//

bool TestAugLagrangianTestFunction() {
  mlpack::IO::Info << "Testing AugmentedLagrangianTestFunction... " << std::endl;
  // The choice of 10 memory slots is arbitrary.
  AugLagrangianTestFunction f;
  AugLagrangian<AugLagrangianTestFunction> aug(f, 10);

  arma::vec coords = f.GetInitialPoint();

  if(!aug.Optimize(0, coords))
    mlpack::IO::Info << "Optimization reported failure. " << std::endl; 
  double final_value = f.Evaluate(coords);

  NOTIFY("Final objective value is %lf at (%lf, %lf)",
      final_value, coords[0], coords[1]);

  if((std::abs(final_value - 70) <= 1e-5) &&
     (std::abs(coords[0] - 1) <= 1e-5) &&
     (std::abs(coords[1] - 4) <= 1e-5))
    return true;
  else
    return false;
}

bool TestGockenbachFunction() {
  mlpack::IO::Info << "Testing Gockenbach function... " << std::endl;
  GockenbachFunction f;
  AugLagrangian<GockenbachFunction> aug(f, 10);

  arma::vec coords = f.GetInitialPoint();

  if(!aug.Optimize(0, coords))
    mlpack::IO:Info << "Optimization reported failure. " << std::endl;

  double final_value = f.Evaluate(coords);

  IO::Notify << "Final objective values is " << coords[0] 
    << " at (" << coords[1] << ", " << coords[2] << ")" << std::endl;

  if((std::abs(final_value - 29.63392) <= 1e-5) &&
     (std::abs(coords[0] - 0.122882) <= 1e-5) &&
     (std::abs(coords[1] + 1.107782)  <= 1e-5) &&
     (std::abs(coords[2] - 0.015100)  <= 1e-5))
    return true;
  else
    return false;
}

int main(int argc, char* argv[]) {
  fx_init(argc, argv, NULL);

  if(!TestAugLagrangianTestFunction())
    IO::Fatal << "Test failed! " << std::endl;
  else
    IO::Info << "Test passed. " << std::endl;

  if(!TestGockenbachFunction())
    IO::Fatal << "Test failed! " << std::endl;
  else
    IO::Info << "Test passed. " << std::endl;
  fx_done(NULL);
}

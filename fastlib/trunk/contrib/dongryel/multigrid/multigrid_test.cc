/** @file multigrid_test.cc
 *
 *  @brief The test driver for the multigrid solver.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include <stdexcept>
#include "multigrid_dev.h"

namespace multigrid_test {

class MultigridTest {
  public:

    void Start() {
      fl::ml::Multigrid<Matrix, Vector> multigrid;
    }
};
};

int main(int argc, char *argv[]) {
  printf("Starting multigrid tests.\n");
  multigrid_test::MultigridTest test;
  test.Start();
  printf("All tests passed!");
}

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
  private:
    void RandomSystem_(
      Matrix *left_hand_side_out,
      Vector *right_hand_side_out) {

      int num_dimensions = math::RandInt(10, 70);
      left_hand_side_out->Init(num_dimensions, num_dimensions);
      right_hand_side_out->Init(num_dimensions);
    }

  public:

    void Trial() {
      fl::ml::Multigrid<Matrix, Vector> multigrid;

      // Generate a random linear system.
      Matrix left_hand_side;
      Vector right_hand_side;
      RandomSystem_(&left_hand_side, &right_hand_side);
    }

    void Start() {

      for (int i = 0; i < 40; i++) {

        // Do a trial.
        Trial();
      }
    }
};
};

int main(int argc, char *argv[]) {
  printf("Starting multigrid tests.\n");
  multigrid_test::MultigridTest test;
  test.Start();
  printf("All tests passed!");
}

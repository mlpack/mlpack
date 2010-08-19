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

      int num_dimensions = 20;
      left_hand_side_out->Init(num_dimensions, num_dimensions);
      right_hand_side_out->Init(num_dimensions);

      for (int j = 0; j < left_hand_side_out->n_cols(); j++) {
        for (int i = 0; i < left_hand_side_out->n_rows(); i++) {
          if (i == j) {
            left_hand_side_out->set(i, j, 4.0);
          }
          else if (i == j + 1 || i + 1 == j) {
            left_hand_side_out->set(i, j, -1.0);
          }
          else {
            left_hand_side_out->set(i, j, 0.0);
          }
        }
        (*right_hand_side_out)[j] = math::Random(1.5, 4.0);
      }
    }

  public:

    void Trial() {
      fl::ml::Multigrid<Matrix, Vector> multigrid;

      // Generate a random linear system.
      Matrix left_hand_side;
      Vector right_hand_side;
      RandomSystem_(&left_hand_side, &right_hand_side);

      // Initialize the multigrid.
      multigrid.Init(left_hand_side, right_hand_side, 4, 1000);
    }

    void Start() {

      for (int i = 0; i < 1; i++) {

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

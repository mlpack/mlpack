#include "fastlib/fastlib.h"
#include "fastlib/base/test.h"
#include "kaczmarz_method.h"

class LinearAlgebraTest {

 public:
  void Init(fx_module *module) {
    module_ = module;
  }

  void TestKaczmarzMethod() {
    Matrix linear_system;
    Vector right_hand_side;
    int num_rows = 3;
    int num_cols = 3;
    linear_system.Init(num_rows, num_cols);
    right_hand_side.Init(num_cols);
    linear_system.set(0, 0, 2);
    linear_system.set(0, 1, 1);
    linear_system.set(0, 2, -1);
    linear_system.set(1, 0, -3);
    linear_system.set(1, 1, 4);
    linear_system.set(1, 2, 2);
    linear_system.set(2, 0, 2);
    linear_system.set(2, 1, -1);
    linear_system.set(2, 2, 1);
    right_hand_side[0] = 21;
    right_hand_side[1] = 1;
    right_hand_side[2] = 17;
    
    Vector solution;
    KaczmarzMethod::SolveTransInit(linear_system, right_hand_side, 
				   &solution);

    linear_system.PrintDebug();
    right_hand_side.PrintDebug();
    solution.PrintDebug();

    // Check whether the solution is close enough to the real one.
    Vector product_between_matrix_and_solution;
    la::MulInit(solution, linear_system,
		&product_between_matrix_and_solution);
    
    product_between_matrix_and_solution.PrintDebug();
  }

  void TestAll() {
    TestKaczmarzMethod();
    NOTIFY("[*] All tests passed !!");
  }  

  void Destruct() {
  }

 private:
  fx_module *module_;
};

int main(int argc, char *argv[]) {

  fx_module *module = fx_init(argc, argv, NULL);
  LinearAlgebraTest test;
  test.Init(module);
  test.TestAll();
  fx_done(module);
  return 0;
}

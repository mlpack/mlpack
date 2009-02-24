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
    int num_rows = 10;
    int num_cols = 20;
    linear_system.Init(num_rows, num_cols);
    right_hand_side.Init(num_rows);
    for(index_t i = 0; i < num_cols; i++) {
      for(index_t j = 0; j < num_rows; j++) {
	linear_system.set(j, i, math::Random());
      }
    }
    for(index_t j = 0; j < num_rows; j++) {
      right_hand_side[j] = math::Random();
    }

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

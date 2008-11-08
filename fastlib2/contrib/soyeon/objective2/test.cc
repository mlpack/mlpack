#include "objective2.h"

class ObjectiveTest {
 public:
  void Init(fx_module *module) {
    module_= module;
  }
  void Destruct();
  void Test1() {
    objective.Init(module_);
		double dummy_objective;
    Matrix x;
    objective.ComputeObjective(x, &dummy_objective );
		NOTIFY("The objective is %lg", dummy_objective);
  }
  void TestAll() {
    Test1();
		
  }

 private:
  Objective objective;
  fx_module *module_;

};



int main(int argc, char *argv[]) {
  fx_module *module=fx_init(argc, argv, NULL);
  ObjectiveTest test;
  test.Init(module);
  test.TestAll();
  fx_done(module);
}

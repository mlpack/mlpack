#include "objective2.h"
#include <iostream.h>

class ObjectiveTest {
 public:
  void Init(fx_module *module) {
    module_= module;
  }
  void Destruct();
  void Test1() {
    objective.Init(module_);
		double dummy_objective;
    //Matrix x;
    objective.ComputeObjective(&dummy_objective );
		NOTIFY("The objective is %g", dummy_objective);

		NOTIFY("gradient calculation start");
		Vector gradient;
		//gradient.Init(num_of_betas_);
		objective.ComputeGradient(&gradient);
		//printf("The objective is %g", dummy_objective);
		cout<<"gradient test "<<gradient[0]<<endl;
		NOTIFY("gradient calculation end");
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

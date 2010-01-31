#include "objective2.h"
#include <iostream.h>

class ObjectiveTest {
 public:
  void Init(fx_module *module) {
    module_= module;
  }
  void Destruct();
  void Test_objective2() {
    objective.Init(module_);

		Vector current_parameter;
		current_parameter.Init(5);
		for(index_t i=0; i<5; i++){
		current_parameter[i]=0.1+i;
		}



		double dummy_objective;
    //Matrix x;
    objective.ComputeObjective(current_parameter, &dummy_objective );
		NOTIFY("The objective is %g", dummy_objective);

		NOTIFY("Gradient calculation starts");
		Vector dummy_gradient;
		//gradient.Init(num_of_betas_);
		objective.ComputeGradient(current_parameter, &dummy_gradient);
		//printf("The objective is %g", dummy_objective);
		cout<<"Gradient vector: ";
		for (index_t i=0; i<dummy_gradient.length(); i++)
		{
			cout<<dummy_gradient[i]<<" ";
		}
		cout<<endl;
		
		NOTIFY("Gradient calculation ends");
		
		NOTIFY("Exact hessian calculation starts");
		Matrix dummy_hessian;
		objective.ComputeHessian(current_parameter, &dummy_hessian);
		cout<<"Hessian matrix: "<<endl;

		for (index_t j=0; j<dummy_hessian.n_rows(); j++){
			for (index_t k=0; k<dummy_hessian.n_cols(); k++){
				cout<<dummy_hessian.get(j,k) <<"  ";
			}
			cout<<endl;
		}
		NOTIFY("Exact hessian calculation ends");
		
  }

	//void Test_sampling() {
	//}


  void TestAll() {
    Test_objective2();
		//Test_sampling();
		
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

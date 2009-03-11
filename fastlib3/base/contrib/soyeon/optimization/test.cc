#include "objective2.h"
#include "optimization.h"
#include <iostream.h>

class OptimizationTest{
public:
	void Init(fx_module *module) {
		module_=module;
	}
	void Destruct();
	void Test_optimization(){
		optimization.Init(module_);
		double current_radius=2;
		Vector current_gradient;
		current_gradient.Init(3);
		current_gradient[0]=0.2;
		current_gradient[1]=0.3;
		current_gradient[2]=0.8;

		Matrix current_hessian;
		current_hessian.Init(3,3);
		current_hessian.set(0,0,1);
		current_hessian.set(0,1,0.5);
		current_hessian.set(0,2,0.7);
		current_hessian.set(1,0,0.5);
		current_hessian.set(1,1,0.9);
		current_hessian.set(1,2,0.8);
		current_hessian.set(2,0,0.7);
		current_hessian.set(2,1,0.8);
		current_hessian.set(2,2,1.3);

		Vector dummy_p;
		double dummy_delta_m;

		optimization.ComputeDoglegDirection(current_radius, 
																current_gradient,
																current_hessian,
																&dummy_p,
																&dummy_delta_m);
		cout<<"p="<<" ";
		for(index_t i=0; i<dummy_p.length(); i++){
			cout<<dummy_p[i]<<" ";
		}
		cout<<endl;

		cout<<"delta_m="<<dummy_delta_m<<endl;
		
    Vector dummy_p2;
		double dummy_delta_m2;
		optimization.ComputeSteihaugDirection(current_radius, 
																current_gradient,
																current_hessian,
																&dummy_p2,
																&dummy_delta_m2);

		cout<<"p2="<<" ";
		for(index_t i=0; i<dummy_p2.length(); i++){
			cout<<dummy_p2[i]<<" ";
		}
		cout<<endl;

		cout<<"delta_m2="<<dummy_delta_m2<<endl;
		


	}

	void TestAll() {
    Test_optimization();
		//Test_Sampling();
				
  }

 private:
  Optimization optimization;
  fx_module *module_;

};



int main(int argc, char *argv[]) {
  fx_module *module=fx_init(argc, argv, NULL);
  OptimizationTest test;
  test.Init(module);
  test.TestAll();
  fx_done(module);
}


  

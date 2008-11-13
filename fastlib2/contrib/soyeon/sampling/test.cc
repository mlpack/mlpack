#include "sampling.h"
#include <iostream>

class SamplingTest {
	public:
		void Init(fx_module *module) {
			module_=module;
		}
		void Destruct();
		void Test_sampling(){
			sampling.Init(module_);
			ArrayList<Matrix> current_added_first_stage_x;
			ArrayList<Matrix> current_added_second_stage_x;
			ArrayList<Matrix> current_added_unknown_x_past;
			ArrayList<index_t> current_added_first_stage_y;
			Vector current_ind_unknown_x;
	
			sampling.ExpandSubset(initial_percent_sample_, &current_added_first_stage_x,
							&current_added_second_stage_x, &current_added_unknown_x_past, 
							&current_added_first_stage_y, &current_ind_unknown_x);



		}

		void TestAll() {
			Test_sampling();
		}

	private:
		Sampling sampling;
		fx_module *module_;
	
};

int main(int argc, char *argv[]) {
  fx_module *module=fx_init(argc, argv, NULL);
  ObjectiveSampling test;
  test.Init(module);
  test.TestAll();
  fx_done(module);
}





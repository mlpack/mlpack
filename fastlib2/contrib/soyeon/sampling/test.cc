#include "sampling.h"
#include "iostream.h"

class SamplingTest {
	public:
		void Init(fx_module *module) {
			module_=module;
		}
		void Destruct();
		void Test_sampling(){
			sampling.Init(module_);
			sampling.Shuffle_();

			ArrayList<Matrix> current_added_first_stage_x;
			ArrayList<Matrix> current_added_second_stage_x;
			ArrayList<Matrix> current_added_unknown_x_past;
			ArrayList<index_t> current_added_first_stage_y;
			Vector current_ind_unknown_x;
	
			sampling.ExpandSubset(50, &current_added_first_stage_x,
							&current_added_second_stage_x, &current_added_unknown_x_past, 
							&current_added_first_stage_y, &current_ind_unknown_x);
			

			ArrayList<Matrix> new_added_first_stage_x;
			ArrayList<Matrix> new_added_second_stage_x;
			ArrayList<Matrix> new_added_unknown_x_past;
			ArrayList<index_t> new_added_first_stage_y;
			Vector new_ind_unknown_x;

			sampling.ExpandSubset(50, &new_added_first_stage_x,
							&new_added_second_stage_x, &new_added_unknown_x_past, 
							&new_added_first_stage_y, &new_ind_unknown_x);
			
			ArrayList<Matrix> a;
			a.Copy(new_added_first_stage_x);
			cout<<"a="<<a[0].get(0,0)<<endl;
			cout<<"a="<<a[0].get(0,1)<<endl;
			cout<<"new_added_first_stage_x="<<new_added_first_stage_x[0].get(0,0)<<endl;
			cout<<"new_added_first_stage_x="<<new_added_first_stage_x[0].get(0,1)<<endl;

			a.Destruct();
			a.Copy(current_added_first_stage_x);
			cout<<"a="<<a[0].get(0,0)<<endl;
			cout<<"a="<<a[0].get(0,1)<<endl;
			cout<<"current_added_first_stage_x="<<current_added_first_stage_x[0].get(0,0)<<endl;
			cout<<"current_added_first_stage_x="<<current_added_first_stage_x[0].get(0,1)<<endl;








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
  SamplingTest test;
  test.Init(module);
  test.TestAll();
  fx_done(module);
}





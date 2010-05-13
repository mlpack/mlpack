#include "sampling.h"
#include "iostream.h"

class SamplingTest {
	public:
		void Init(fx_module *module) {
			module_=module;
		}
		void Destruct();
		//void Test_sampling(double *num_of_people, Vector *ind_unknown_x){
		void Test_sampling(double percent_added_sample){
			int num_of_people;
			Vector ind_unknown_x;
      double initial_percent_sample;
			Vector initial_parameter;
			
			sampling.Init(module_, &num_of_people, &ind_unknown_x, 
                  &initial_percent_sample,
									&initial_parameter);

			NOTIFY("Number of people in dataset is %d", num_of_people);
			NOTIFY("Shuffling");
			sampling.Shuffle_();
      NOTIFY("Sampling starts with %f percent of population", initial_percent_sample);
			NOTIFY("Starting Point:");
			for(index_t i=0; i<initial_parameter.length(); i++){
				cout<<initial_parameter[i]<<" ";
			}
			cout<<endl;
			
			ArrayList<Matrix> current_added_first_stage_x;
			current_added_first_stage_x.Init();

			ArrayList<Matrix> current_added_second_stage_x;
			current_added_second_stage_x.Init();

			ArrayList<Matrix> current_added_unknown_x_past;
			current_added_unknown_x_past.Init();

			ArrayList<index_t> current_added_first_stage_y;
			current_added_first_stage_y.Init();

			int count=0;
			int sample_size=0;
			double current_percent_added_sample=0;
			
			while(sample_size<num_of_people){
				current_percent_added_sample=percent_added_sample;
				
				
				sampling.ExpandSubset(current_percent_added_sample, &current_added_first_stage_x,
							&current_added_second_stage_x, &current_added_unknown_x_past, 
							&current_added_first_stage_y);
				count+=1;
				cout<<"current_added_first_stage_x.size()="<<current_added_first_stage_x.size()<<endl;
				sample_size=current_added_first_stage_x.size();
			}
			cout<<"count="<<count<<endl;

			//cout<<"initial_percent_sample="<<initial_percent_sample_<<endl;

			//Vector current_ind_unknown_x;
			//current_ind_unknown_x.Init();

	
			/*sampling.ExpandSubset(50, &current_added_first_stage_x,
							&current_added_second_stage_x, &current_added_unknown_x_past, 
							&current_added_first_stage_y, &current_ind_unknown_x);
			*/
			
			/*sampling.ExpandSubset(percent_added_sample, &current_added_first_stage_x,
							&current_added_second_stage_x, &current_added_unknown_x_past, 
							&current_added_first_stage_y);
			*/
			/*cout<<"sampling result 1:";
			for(index_t i=0; i<current_added_first_stage_x.size(); i++) {
			cout<<"current_added_first_stage_x["<<i<<"]"<<endl;
				for(index_t j=0; j<current_added_first_stage_x[i].n_rows(); j++){
					for(index_t k=0; k<current_added_first_stage_x[i].n_cols(); k++) {
						cout<<current_added_first_stage_x[i].get(j,k)<<" ";
					}
					cout<<endl;
				}
				cout<<endl;		
			}

			for(index_t i=0; i<current_added_second_stage_x.size(); i++) {
			cout<<"current_added_second_stage_x["<<i<<"]"<<endl;
				for(index_t j=0; j<current_added_second_stage_x[i].n_rows(); j++){
					for(index_t k=0; k<current_added_second_stage_x[i].n_cols(); k++) {
						cout<<current_added_second_stage_x[i].get(j,k)<<" ";
					}
					cout<<endl;
				}
				cout<<endl;		
			}
			*/


			/*sampling.ExpandSubset(50, &current_added_first_stage_x,
							&current_added_second_stage_x, &current_added_unknown_x_past, 
							&current_added_first_stage_y);

			cout<<"sampling result 2:";
			for(index_t i=0; i<current_added_first_stage_x.size(); i++) {
			cout<<"current_added_first_stage_x["<<i<<"]"<<endl;
				for(index_t j=0; j<current_added_first_stage_x[i].n_rows(); j++){
					for(index_t k=0; k<current_added_first_stage_x[i].n_cols(); k++) {
						cout<<current_added_first_stage_x[i].get(j,k)<<" ";
					}
					cout<<endl;
				}
				cout<<endl;		
			}

			for(index_t i=0; i<current_added_second_stage_x.size(); i++) {
			cout<<"current_added_second_stage_x["<<i<<"]"<<endl;
				for(index_t j=0; j<current_added_second_stage_x[i].n_rows(); j++){
					for(index_t k=0; k<current_added_second_stage_x[i].n_cols(); k++) {
						cout<<current_added_second_stage_x[i].get(j,k)<<" ";
					}
					cout<<endl;
				}
				cout<<endl;		
			}
*/


			/*sampling.ExpandSubset(50, &current_added_first_stage_x,
							&current_added_second_stage_x, &current_added_unknown_x_past, 
							&current_added_first_stage_y, &current_ind_unknown_x);
							*/
			/*cout<<"after second sampling="<<endl;
			cout<<"size="<<current_added_first_stage_x.size()<<endl;
			cout<<"current_added_first_stage_x0="<<current_added_first_stage_x[0].get(0,0)<<endl;
			cout<<"current_added_first_stage_x1="<<current_added_first_stage_x[1].get(0,1)<<endl;
			cout<<"current_added_first_stage_x2="<<current_added_first_stage_x[2].get(0,1)<<endl;
*/

		}

		void TestAll() {
			//double num_of_people;
			//Vector ind_unknown_x;
			Test_sampling(30);
			//cout<<"num people="<<num_of_people<<endl;
		}

	private:
		Sampling sampling;
		fx_module *module_;
	
};

int main(int argc, char *argv[]) {
  fx_module *module=fx_init(argc, argv, NULL);
	

  SamplingTest test;
  test.Init(module);
	//cout<<"initial_percent_sample_="<<initial_percent_sample_<<endl;
  test.TestAll();
  fx_done(module);
}





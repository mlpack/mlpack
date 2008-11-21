#include "sampling.h"
#include "objective2.h"
#include "optimization.h"
#include <iostream.h>

/*class DDCMTest {
public:
		void Init(fx_module *module) {
			module_=module;
		}
		void Destruct();
		void Test_sampling(double percent_added_sample) {
		}
		void Test_objective2() {
		}
		void Test_optimization() {
		}

		void TestAll(){
		}

private:
	DDCM ddcm;
	fx_module *module_;
		
};

*/


int main(int argc, char *argv[]) {
  fx_module *module=fx_init(argc, argv, NULL);
	

  Sampling sampling;
	Objective objective;
	Optimization optimization;

	int num_of_people;
	Vector ind_unknown_x;
	double initial_percent_sampling;
	Vector initial_parameter;

  sampling.Init(module, &num_of_people, &ind_unknown_x, 
								&initial_percent_sampling,
								&initial_parameter);

	NOTIFY("Number of people in dataset is %d", num_of_people);
	NOTIFY("Shuffling");
	sampling.Shuffle();
	sampling.Shuffle();
	NOTIFY("Initial sampling percent is %f", initial_percent_sampling);
	int count_init2=0;
	objective.Init2(ind_unknown_x, count_init2);
	count_init2+=1;
	//optimization.Init(module);
	
	ArrayList<Matrix> current_added_first_stage_x;
	current_added_first_stage_x.Init();

	ArrayList<Matrix> current_added_second_stage_x;
	current_added_second_stage_x.Init();

	ArrayList<Matrix> current_added_unknown_x_past;
	current_added_unknown_x_past.Init();

	ArrayList<index_t> current_added_first_stage_y;
	current_added_first_stage_y.Init();

	int sampling_count=0;
	int sample_size=0;
	double current_percent_added_sample=initial_percent_sampling;
	Vector current_parameter;
	double dummy_objective;
	current_parameter.Copy(initial_parameter);
	double current_radius=1.1;
	
	while(sample_size<num_of_people){

		
		sampling_count+=1;
		cout<<"sampling_count="<<sampling_count<<endl;
		sampling.ExpandSubset(current_percent_added_sample, &current_added_first_stage_x,
					&current_added_second_stage_x, &current_added_unknown_x_past, 
					&current_added_first_stage_y);
		//current_percent_added_sample=percent_added_sample;
		//index_t current_num_selected_people=current_percent_added_sample.size();
		objective.Init3(sample_size,
									 current_added_first_stage_x,
									 current_added_second_stage_x,
									 current_added_unknown_x_past, 
									 current_added_first_stage_y);
		
		objective.ComputeObjective(current_parameter, 
															 &dummy_objective);
		NOTIFY("The objective is %g", dummy_objective);

		
		NOTIFY("Gradient calculation starts");
		Vector current_gradient;
		//gradient.Init(num_of_betas_);
		objective.ComputeGradient(current_parameter, &current_gradient);
		//printf("The objective is %g", dummy_objective);
		cout<<"Gradient vector: ";
		for (index_t i=0; i<current_gradient.length(); i++)
		{
			cout<<current_gradient[i]<<" ";
		}
		cout<<endl;
		
		NOTIFY("Gradient calculation ends");
		

		NOTIFY("Exact hessian calculation starts");
		Matrix current_hessian;
		objective.ComputeHessian(current_parameter, &current_hessian);
		cout<<"Hessian matrix: "<<endl;

		for (index_t j=0; j<current_hessian.n_rows(); j++){
			for (index_t k=0; k<current_hessian.n_cols(); k++){
				cout<<current_hessian.get(j,k) <<"  ";
			}
			cout<<endl;
		}
		NOTIFY("Exact hessian calculation ends");

		Vector next_p1;
		double next_delta_m1;
		optimization.ComputeDoglegDirection(current_radius, 
																        current_gradient,
																				current_hessian,
																				&next_p1,
																				&next_delta_m1);
		cout<<"p="<<" ";
		for(index_t i=0; i<next_p1.length(); i++){
			cout<<next_p1[i]<<" ";
		}
		cout<<endl;

		cout<<"delta_m="<<next_delta_m1<<endl;
		/*
    Vector next_p2;
		double next_delta_m2;
		optimization.ComputeSteihaugDirection(current_radius, 
																current_gradient,
																current_hessian,
																&next_p2,
																&next_delta_m2);

		cout<<"p2="<<" ";
		for(index_t i=0; i<next_p2.length(); i++){
			cout<<next_p2[i]<<" ";
		}
		cout<<endl;

		cout<<"delta_m2="<<next_delta_m2<<endl;
		*/

		la::AddTo(next_p1, &current_parameter);
		cout<<"new_parameter=";
		for(index_t i=0; i<current_parameter.length(); i++){
			cout<<current_parameter[i]<<" ";
		}
		cout<<endl;




		cout<<"current_added_first_stage_x.size("<<current_added_first_stage_x.size()<<")"<<endl;
		sample_size=current_added_first_stage_x.size();
		


	}
	cout<<"Total_sampling_count="<<sampling_count<<endl;
	

  



  
  fx_done(module);
}






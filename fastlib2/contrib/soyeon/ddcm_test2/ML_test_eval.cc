#include "MLsampling.h"
#include "MLobjective.h"
#include "optimization.h"
#include <iostream>

using namespace std;

int main(int argc, char *argv[]) {
  fx_module *module=fx_init(argc, argv, NULL);
	cout<<"ML model test"<<endl;

  MLSampling sampling;
	MLObjective objective;
	Optimization optimization;
	
	int num_of_people;
	//Vector ind_unknown_x;
	double initial_percent_sampling;
	Vector initial_parameter;
	Vector opt_x;


  sampling.Init3(module, &num_of_people, 
								&initial_percent_sampling,
								&initial_parameter, &opt_x);
	cout<<"Optimal points:"<<endl;
	
	/*
	//sim9 res7 test2
	initial_parameter[0]=4.86856;
	initial_parameter[1]=-1.36438;
	initial_parameter[2]=0.611558;
	initial_parameter[3]=0.271807;
	*/
	   

	      
   
	for(index_t i=0; i<opt_x.length(); i++){
		cout<<opt_x[i]<<" ";
	}
	cout<<endl;
	NOTIFY("Number of people in TEST dataset is %d", num_of_people);
	sampling.Shuffle2();
	
	int count_init2=0;
	objective.Init2(count_init2);
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

	
	

	int sample_size=0;
	int end_sampling=0;
	//int num_added_sample=0;
	double current_percent_added_sample=initial_percent_sampling;
	
	int num_of_parameter=opt_x.length();
	Vector current_parameter;
	//current_parameter.Init(num_of_parameter);
	current_parameter.Copy(opt_x);
	cout<<"num par="<<num_of_parameter<<endl;
	
  sampling.ExpandSubset(current_percent_added_sample, 
												&current_added_first_stage_x,
												&current_added_first_stage_y);
	objective.Init3(sample_size,
							 current_added_first_stage_x,
							 current_added_first_stage_y);

	double current_sample_size;
	current_sample_size=current_added_first_stage_x.size();


	cout<<"Number of TEST data used="<<current_added_first_stage_x.size()<<endl;
	

  double postponed_prediction_error;
	double choice_prediction_error;

	objective.ComputePredictionError(current_sample_size, 
									  current_parameter,
										current_added_first_stage_y,
										&postponed_prediction_error,
										&choice_prediction_error);


 
  fx_done(module);
}

















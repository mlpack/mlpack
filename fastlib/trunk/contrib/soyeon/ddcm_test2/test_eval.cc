#include "sampling.h"
#include "objective2.h"
#include "optimization.h"
#include <iostream>

using namespace std;

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
	cout<<"Optimal points:"<<endl;
	
	initial_parameter[0]=1.52238;
	initial_parameter[1]=5.55074;
	initial_parameter[2]=1.35623;
	initial_parameter[3]=0.77979;
	/*
	initial_parameter[0]=1.52238;
	initial_parameter[1]=5.55074;
	initial_parameter[2]=1.35623;
	initial_parameter[3]=0.77979;

	initial_parameter[0]=1;
	initial_parameter[1]=4;
	initial_parameter[2]=5;
	initial_parameter[3]=2;
		
	initial_parameter[0]=1.60956;
	initial_parameter[1]=5.96071;
	initial_parameter[2]=5.27698;
	initial_parameter[3]=1.23762;

	initial_parameter[0]=-1.96832;
	initial_parameter[1]=3.23816;
	initial_parameter[2]=2.96586;
	initial_parameter[3]=1.24199;

	//sim9 res3 test
	initial_parameter[0]=1.52046;
	initial_parameter[1]=-2.09877;
	initial_parameter[2]=4.9505;
	initial_parameter[3]=1.84854;

	//sim9 res3 test2
	initial_parameter[0]=1.5178;
	initial_parameter[1]=-2.10037;
	initial_parameter[2]=4.32019;
	initial_parameter[3]=1.63162;

  //sim9 res4 test
	initial_parameter[0]=4.22544;
	initial_parameter[1]=-3.11216;
	initial_parameter[2]=2.65523;
	initial_parameter[3]=0.38569;

	//sim9 res4 test2
	initial_parameter[0]=4.22348;
	initial_parameter[1]=-3.09046;
	initial_parameter[2]=3.229;
	initial_parameter[3]=0.503605;

	//sim9 res4 test3
	initial_parameter[0]=4.21784;
	initial_parameter[1]=-3.08405;
	initial_parameter[2]=3.69841;
	initial_parameter[3]=0.595435;

	//sim9 res4 test4
	initial_parameter[0]=4.21104;
	initial_parameter[1]=-3.08401;
	initial_parameter[2]=4.09364;
	initial_parameter[3]=0.670095;

	//sim9 res4 test4
	initial_parameter[0]=4.21155;
	initial_parameter[1]=-3.08093;
	initial_parameter[2]=4.19198;
	initial_parameter[3]=0.687958;

	//sim9 res5 test
	initial_parameter[0]=3.3367;
	initial_parameter[1]=-2.25195;
	initial_parameter[2]=3.58861;
	initial_parameter[3]=0.0613634;

	//sim9 res6 test
	initial_parameter[0]=3.1008;
	initial_parameter[1]=-2.32536;
	initial_parameter[2]=6.50685;
	initial_parameter[3]=0.444984;


	//sim9 res6 test2
	initial_parameter[0]=3.10166;
	initial_parameter[1]=-2.3239;
	initial_parameter[2]=6.71464;
	initial_parameter[3]=0.470468;

	//sim9 res6 test2
	initial_parameter[0]=4.89444;
	initial_parameter[1]=-1.36876;
	initial_parameter[2]=0.267926;
	initial_parameter[3]=1.64124e-12;

	//sim8 res4 test
	initial_parameter[0]=1.43946;
	initial_parameter[1]=6.11284;
	initial_parameter[2]=2.5596;
	initial_parameter[3]=0.588488;


	//sim8 res2 test
	initial_parameter[0]=1.52238;
	initial_parameter[1]=5.55074;
	initial_parameter[2]=1.35623;
	initial_parameter[3]=0.77979;

	//sim8 res1 test
	initial_parameter[0]=1.39487;
	initial_parameter[1]=5.43331;
	initial_parameter[2]=4.83443;
	initial_parameter[3]=1.57564;*/

	//sim9 res7 test2
	initial_parameter[0]=4.86856;
	initial_parameter[1]=-1.36438;
	initial_parameter[2]=0.611558;
	initial_parameter[3]=0.271807;
	   
   
	      

	   

	      

	  

	   
	   

	   

	   

	   

	      
   
	for(index_t i=0; i<initial_parameter.length(); i++){
		cout<<initial_parameter[i]<<" ";
	}
	cout<<endl;
	NOTIFY("Number of people in TEST dataset is %d", num_of_people);
	sampling.Shuffle();
	
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

	
	

	int sample_size=0;
	int end_sampling=0;
	//int num_added_sample=0;
	double current_percent_added_sample=initial_percent_sampling;
	
	int num_of_parameter=initial_parameter.length();
	Vector current_parameter;
	//current_parameter.Init(num_of_parameter);
	current_parameter.Copy(initial_parameter);
	cout<<"num par="<<num_of_parameter<<endl;
	
  sampling.ExpandSubset(current_percent_added_sample, 
												&current_added_first_stage_x,
												&current_added_second_stage_x, 
												&current_added_unknown_x_past,
												&current_added_first_stage_y);
	objective.Init3(sample_size,
							 current_added_first_stage_x,
							 current_added_second_stage_x,
							 current_added_unknown_x_past, 
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

















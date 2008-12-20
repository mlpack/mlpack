#include "sampling.h"
#include "objective2.h"
#include "optimization.h"
#include <iostream>

using namespace std;

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
	cout<<"Starting points:"<<endl;
	for(index_t i=0; i<initial_parameter.length(); i++){
		cout<<initial_parameter[i]<<" ";
	}
	cout<<endl;
	NOTIFY("Number of people in dataset is %d", num_of_people);
	NOTIFY("Shuffling");
	sampling.Shuffle();
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

	
	

	int sample_size=0;
	int end_sampling=0;
	//int num_added_sample=0;
	double current_percent_added_sample=initial_percent_sampling;
	
	int num_of_parameter=initial_parameter.length();
	Vector current_parameter;
	//current_parameter.Init(num_of_parameter);
	current_parameter.Copy(initial_parameter);
	Vector next_parameter;
	cout<<"num par="<<num_of_parameter<<endl;
	next_parameter.Init(num_of_parameter);
	next_parameter.SetZero();

	
  
	double current_objective;
	double next_objective;
	double p_norm=0;

	//Trust region parameter
	//double max_radius=10;
	double current_radius=0.001;	//initial_radius
	double eta=0.2;
	
	double rho=0; //agreement(ratio)
	
	

	double correction_factor=0;
	Vector current_choice_probability;
	Vector next_choice_probability;

	//for stopping rule
	double error_tolerance=1e-16;
	double zero_tolerance=1e-1;	//for gradient norm

	//error_tolerance*=100000;
	//cout<<"error_tolerance="<<error_tolerance<<endl;
	Vector tpar;
	tpar.Init(current_parameter.length());
	tpar[0]=1;
	tpar[1]=1;
	tpar[2]=2;
	tpar[3]=2;

  cout<<"true parameter:"<<endl;
	for(index_t i=0; i<tpar.length(); i++){
		cout<<tpar[i]<<" ";
	}
	cout<<endl;


  double tobjective;
	objective.ComputeObjective(tpar, 
															 &tobjective);
	cout<<"max objective="<<tobjective<<endl;

	int max_iteration=500;
	int iteration_count=0;

	while(iteration_count<max_iteration){

		
		iteration_count+=1;
		cout<<"iteration_count="<<iteration_count<<endl;

		if(end_sampling==0) {
			//NOTIFY("All data are used");
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
		
		}
		else {
			
			NOTIFY("All data are used");
		}

		cout<<"Number of data used="<<current_added_first_stage_x.size()<<endl;
		objective.ComputeObjective(current_parameter, 
															 &current_objective);
		
		current_objective/=current_added_first_stage_x.size();
		NOTIFY("The objective is %g", current_objective);

		
		//NOTIFY("Gradient calculation starts");
		Vector current_gradient;
		//gradient.Init(num_of_betas_);
		objective.ComputeGradient(current_parameter, &current_gradient);
		//printf("The objective is %g", dummy_objective);
		/*
		cout<<"Gradient vector: ";
		for (index_t i=0; i<current_gradient.length(); i++)
		{
			std::cout<<current_gradient[i]<<" ";
		}
		std::cout<<endl;
		*/

		
		//NOTIFY("Gradient calculation ends");
		
		

		//NOTIFY("Exact hessian calculation starts");
		Matrix current_hessian;
		objective.ComputeHessian(current_parameter, &current_hessian);
		
		cout<<"Hessian matrix: "<<endl;

		/*
		for (index_t j=0; j<current_hessian.n_rows(); j++){
			for (index_t k=0; k<current_hessian.n_cols(); k++){
				cout<<current_hessian.get(j,k) <<"  ";
			}
			cout<<endl;
		}
		*/
		

		Matrix current_inverse_hessian;
		if( !PASSED(la::InverseInit(current_hessian, &current_inverse_hessian)) ) {
			NOTIFY("Current hessian matrix is not invertible!");
		}
		else{
			cout<<"Diagonal of inverse hessian: ";
			for(index_t i=0; i<current_inverse_hessian.n_rows(); i++){
				cout<<current_inverse_hessian.get(i,i)<<" ";
			}
			cout<<endl;
		}



		Vector current_p;
		double current_delta_m;
		Vector next_parameter;
		double new_radius;
		//NOTIFY("Exact hessian calculation ends");

		/*
		optimization.ComputeDerectionUnderConstraints(current_radius, 
																					current_gradient,
																					current_hessian,
																					current_parameter,
																					&current_p,
																					&current_delta_m,
																					&next_parameter,
																					&new_radius);
		*/
		

		
		//Scaled version
		optimization.ComputeScaledDerectionUnderConstraints(current_radius, 
																					current_gradient,
																					current_hessian,
																					current_parameter,
																					&current_p,
																					&current_delta_m,
																					&next_parameter,
																					&new_radius);

		
		


		current_radius=new_radius;

/*	
		double constraints_check=1;
			
		//while(constraints_check>0){

			
		optimization.ComputeDoglegDirection(current_radius, 
																					current_gradient,
																					current_hessian,
																					&current_p,
																					&current_delta_m);
		//double p_norm=0;
			

		
		//cout<<"p="<<" ";
		//for(index_t i=0; i<current_p.length(); i++){
		//	cout<<current_p[i]<<" ";
		//}
		//cout<<endl;
		
		cout<<"delta_m="<<current_delta_m<<endl;

		
		la::AddOverwrite(current_p, current_parameter, &next_parameter);

		if( next_parameter[next_parameter.length()-2]>0 && 
			next_parameter[next_parameter.length()-1]>0 ) {
			NOTIFY("Satisfy Constraints");
			constraints_check=-1;
		}
		else{
			cout<<"bed_new_parameter=";
			for(index_t i=0; i<next_parameter.length(); i++){
				cout<<next_parameter[i]<<" ";
			}
			cout<<endl;
				
		  NOTIFY("Constraint Violation...Shrink Trust region Radius...");
		  current_radius=0.75*current_radius;
			
			NOTIFY("Projection");
			if(next_parameter[next_parameter.length()-2]<=0) {
				NOTIFY("p is negative");
				next_parameter[next_parameter.length()-2]=0;
			}
			else if(next_parameter[next_parameter.length()-1]<=0) {
				NOTIFY("q is negative");
				next_parameter[next_parameter.length()-1]=0;
			}
			
							
		}	//else

		
*/
		//Scaling
		//Vector nonscaled_current_p;
		//nonscaled_current_p.Init(current_p.length());
		//nonscaled_current_p.CopyValues(current_p);
		//la::MulOverwrite(current_inverse_hessian, nonscaled_current_p, &current_p);
		//current_inverse_hessian
				
		p_norm=sqrt(la::Dot(current_p, current_p));
		 
		cout<<"new_parameter=";
		for(index_t i=0; i<next_parameter.length(); i++){
			cout<<next_parameter[i]<<" ";
		}
		cout<<endl;

		objective.ComputeObjective(next_parameter, 
														 &next_objective);
		next_objective/=current_added_first_stage_x.size();
		NOTIFY("The Next objective is %g", next_objective);

	



		//cout<<"current_added_first_stage_x.size("<<current_added_first_stage_x.size()<<")"<<endl;
		//update sample size
		

		if(end_sampling==0){	//if all data are not used yet
			//num_added_sample = current_added_first_stage_x.size()-sample_size;
			sample_size=current_added_first_stage_x.size();
			//NOTIFY("Number of added sample= %d", num_added_sample);
			
			//sampling error calculation
			correction_factor=(num_of_people-sample_size)/(num_of_people-1.0);
			//cout<<"correction factor:"<<correction_factor<<endl;
			
			objective.ComputeChoiceProbability(current_parameter, 
																				 &current_choice_probability);
			objective.ComputeChoiceProbability(next_parameter, 
																				 &next_choice_probability);
			
			double sampling_error=0;
			for(index_t n=0; n<sample_size; n++){
				sampling_error+= pow(((current_choice_probability[n]-next_choice_probability[n])-(current_objective-next_objective) ),2);
			}
			sampling_error*=(correction_factor/(sample_size*(sample_size-1)));
			cout<<"sampling_error="<<sampling_error<<endl;
			
			if(current_delta_m<0.5*sampling_error){
				current_percent_added_sample=(0.5*sampling_error/current_delta_m)*(0.5*sampling_error/current_delta_m);
				current_percent_added_sample*=100.0;
				//cout<<"percent_added_sample="<<current_percent_added_sample<<endl;
			}

		}	//if

		
		/*
		if(sample_size==num_of_people &&end_sampling==0){
			end_sampling+=1;
			NOTIFY("All data are used");
		}
		*/

		
				
		//agreement rho calculation
		rho=+1.0*(current_objective-next_objective)/(current_delta_m);
		cout<<"rho= "<<rho<<endl;
		if(rho>eta){
			current_parameter.CopyValues(next_parameter);
			NOTIFY("Accepting the step...");
		}
		
		//radius update
		optimization.TrustRadiusUpdate(rho, p_norm, &current_radius);

		if(sample_size==num_of_people){
			end_sampling+=1;
			double gradient_norm;
			la::Scale(1.0/num_of_people, &current_gradient);
			gradient_norm = sqrt(la::Dot(current_gradient, current_gradient));
			
			cout<<"gradient_norm="<<gradient_norm<<endl;

			if(gradient_norm<zero_tolerance){
				NOTIFY("Gradient norm is small enough...Exit...");
				break;
			}
			
			//NOTIFY("All data are used");
		}

	}
	
		



	
	cout<<"Total_iteration_count="<<iteration_count<<endl;
	NOTIFY("Final solution: ");
	for(index_t i=0; i<current_parameter.length(); i++) {
		cout<<current_parameter[i]<<" ";
	}
	cout<<endl;

	//Compute Variance of the estimates -H^{-1}
	Matrix final_hessian;
	objective.ComputeHessian(current_parameter, &final_hessian);
	Matrix inverse_hessian;
	if( !PASSED(la::InverseInit(final_hessian, &inverse_hessian)) ) {
		NOTIFY("Final hessian matrix is not invertible!");
	}
	else{
		//la::Scale(-1.0, &inverse_hessian);

		cout<<"Diagonal of inverse final hessian: ";
		for(index_t i=0; i<inverse_hessian.n_rows(); i++){
			cout<<inverse_hessian.get(i,i)<<" ";
		}
		cout<<endl;


		index_t n=inverse_hessian.n_rows();
		Vector estimates_variance;
		estimates_variance.Init(n);
		for(index_t i=0; i<n; i++) {
			estimates_variance[i]=inverse_hessian.get(i,i);
		}


		//linalg__private::DiagToVector(inverse_hessian, &estimates_variance);
		
		cout<<"Variance of etimates: ";
		for(index_t i=0; i<estimates_variance.length(); i++) {
			cout<<estimates_variance[i]<<" ";

		}
		cout<<endl;

	}

	
	

 
  fx_done(module);
}










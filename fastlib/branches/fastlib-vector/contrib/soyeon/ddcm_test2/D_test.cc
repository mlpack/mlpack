#include "Dsampling.h"
#include "Dobjective2.h"
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

	cout<<"D-ddcm model training"<<endl;
	

  DSampling sampling;
	DObjective objective;
	Optimization optimization;

	int num_of_people;
	Vector ind_unknown_x;
	double initial_percent_sampling;
	Vector initial_parameter;


  sampling.Init2(module, &num_of_people, &ind_unknown_x, 
								&initial_percent_sampling,
								&initial_parameter);
	cout<<"Starting points:"<<endl;
	for(index_t i=0; i<initial_parameter.length(); i++){
		cout<<initial_parameter[i]<<" ";
	}
	cout<<endl;
	//NOTIFY("Number of people in dataset is %d", num_of_people);
	cout<<"Number of people in dataset is "<<num_of_people<<endl;
	//num_of_people=5000;
	//cout<<"Number of people in dataset is "<<num_of_people<<endl;
	//NOTIFY("Shuffling");
	cout<<"Shuffling"<<endl;
	//sampling.Shuffle();
	//cout<<"Shuffling2"<<endl;
	sampling.Shuffle2();
	//sampling.Shuffle();
	//NOTIFY("Initial sampling percent is %f", initial_percent_sampling);
	cout<<"Initial sampling percent is "<<initial_percent_sampling<<endl;
		
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
	double current_radius=0.1;	//initial_radius
	//double eta=0.25;
	//double eta=0.0001;
	double eta=0.2;
	
	double rho=0; //agreement(ratio)
	
	double old_objective=0;

	double correction_factor=0;
	Vector current_choice_probability;
	Vector next_choice_probability;

	//for stopping rule
	double error_tolerance=1e-16;
	//double zero_tolerance=0.0001;	//for gradient norm 10^-5?
	double zero_tolerance=0.001;

	//error_tolerance*=100000;
	//cout<<"error_tolerance="<<error_tolerance<<endl;
	
	/*
	Vector tpar;
	tpar.Init(current_parameter.length());
	
	tpar[0]=1;
	tpar[1]=4;
	tpar[2]=5;
	tpar[3]=2;
	

	tpar[0]=-1;
	tpar[1]=2;
	tpar[2]=3;
	tpar[3]=1;

	tpar[0]=0.8;
	tpar[1]=-1.2;
	tpar[2]=3;
	tpar[3]=1;

	tpar[0]=2;
	tpar[1]=-1.5;
	tpar[2]=4;
	tpar[3]=0.8;

	tpar[0]=3;
	tpar[1]=-1;
	tpar[2]=4;
	tpar[3]=1;

	


  
	cout<<"true parameter:"<<endl;
	for(index_t i=0; i<tpar.length(); i++){
		cout<<tpar[i]<<" ";
	}
	cout<<endl;
  */
	
	//hessian update
	Vector diff_gradient;
	diff_gradient.Init(num_of_parameter);
	diff_gradient.SetZero();

	Vector diff_par;
	diff_par.Init(num_of_parameter);
	diff_par.SetZero();

	Matrix temp1; //H*s*s'*H
	temp1.Init(num_of_parameter, num_of_parameter);
	temp1.SetZero();

	Matrix Hs;
	Hs.Init(num_of_parameter, 1);

	double scale_temp1;
	scale_temp1=0;

	Matrix temp2; //yy'/s'y
	temp2.Init(num_of_parameter, num_of_parameter);

	Matrix updated_hessian;
	updated_hessian.Init(num_of_parameter, num_of_parameter);
	updated_hessian.SetZero();





  //iteration
  int max_iteration=100;
	int iteration_count=0;

	/*
	Matrix current_hessian;		
	current_hessian.Init(num_of_parameter, num_of_parameter);
	current_hessian.SetZero();
	for(index_t i=0; i<current_hessian.n_rows(); i++){
		current_hessian.set(i,i,1);
	}
	la::Scale(-1.0, &current_hessian);
	*/

	Matrix current_hessian;	

	



	//objective.ComputeHessian(current_sample_size, current_parameter, &current_hessian);

	while(iteration_count<max_iteration){

		
		iteration_count+=1;
		cout<<"iteration_count="<<iteration_count<<endl;
		//cout<<"end_sampling="<<end_sampling<<endl;

		if(end_sampling==0) {
			//NOTIFY("All data are NOT used");
		sampling.ExpandSubset(current_percent_added_sample, &current_added_first_stage_x,
					&current_added_second_stage_x, &current_added_unknown_x_past, 
					&current_added_first_stage_y);

		//cout<<"sampling 1 done"<<endl;
		//current_percent_added_sample=percent_added_sample;
		//index_t current_num_selected_people=current_percent_added_sample.size();
		objective.Init3(sample_size,
							 current_added_first_stage_x,
							 current_added_second_stage_x,
							 current_added_unknown_x_past, 
							 current_added_first_stage_y);

		//cout<<"objective Init3"<<endl;


		
		}
		else {
			
			//NOTIFY("All data are used");
			cout<<"All data are used"<<endl;
			//old_objective=current_objective;
	
		}
		double current_sample_size;
		current_sample_size=current_added_first_stage_x.size();


		cout<<"Number of data used="<<current_added_first_stage_x.size()<<endl;
		
		/*
		cout<<"current_parameter part1"<<endl;
		for(index_t i=0; i<current_parameter.length(); i++){
			cout<<current_parameter[i]<<" "<<endl;
		}
		*/


    		//double current_objective;
		//current_objective=0;
		objective.ComputeObjective(current_sample_size, current_parameter, 
															 &current_objective);
		
		//current_objective/=current_added_first_stage_x.size();
		NOTIFY("The objective is %g", current_objective);
		cout<<"The objective is "<<current_objective<<endl;
		
		
		cout<<"current_sample_size="<<current_sample_size<<endl;

	

		//NOTIFY("Gradient calculation starts");
		
		

		/*
		
		double tobjective;
		tobjective=0;
		objective.ComputeObjective(current_sample_size, tpar, 
															 &tobjective);
		
		//tobjective/=current_added_first_stage_x.size();
		cout<<"max objective="<<tobjective<<endl;
    
			
    		Vector opt_gradient;
		//gradient.Init(num_of_betas_);
		objective.ComputeGradient(current_sample_size, tpar, &opt_gradient);
		
		//cout<<"current_added_first_stage_x.size()="<<current_added_first_stage_x.size()<<endl;
		//la::Scale(1.0/current_added_first_stage_x.size(), &opt_gradient);
		
		cout<<"Gradient vector at true par: ";
		for (index_t i=0; i<opt_gradient.length(); i++)
		{
			cout<<opt_gradient[i]<<" ";
		}
		cout<<endl;
		
		double opt_gradient_norm;
		opt_gradient_norm = sqrt(la::Dot(opt_gradient, opt_gradient));
		cout<<"gradient_norm at true par="<<opt_gradient_norm<<endl;

		

		*/

			

		Vector current_gradient;
		//gradient.Init(num_of_betas_);
		/*
		cout<<"test current_parameter"<<endl;
		for(index_t i=0; i<current_parameter.length(); i++){
			cout<<current_parameter[i]<<" ";
		}
		cout<<endl;
    */

		objective.ComputeGradient(current_sample_size, current_parameter, &current_gradient);
		//Vector current_gradient2;
		//objective.ComputeGradient(current_sample_size, tpar, &current_gradient2);
		
		//la::Scale(1.0/current_added_first_stage_x.size(), &current_gradient);
		
		//printf("The objective is %g", dummy_objective);
		
		cout<<"Gradient vector: ";
		for (index_t i=0; i<current_gradient.length(); i++)
		{
			std::cout<<current_gradient[i]<<" ";
		}
		std::cout<<endl;

    /*
		Vector approx_gradient;
		objective.CheckGradient(current_sample_size, current_parameter, &approx_gradient);

		cout<<"Approximated Gradient vector: ";
		for (index_t i=0; i<current_gradient.length(); i++)
		{
			cout<<approx_gradient[i]<<" ";
		}
		cout<<endl;
		*/
				
		

				

		//NOTIFY("Gradient calculation ends");
			cout<<"current_sample_size="<<current_sample_size<<endl;
			cout<<"num_of_people="<<num_of_people<<endl;
			//if(sample_size==num_of_people){
			if(current_sample_size>=num_of_people){
			end_sampling+=1;
			double gradient_norm;
			//la::Scale(1.0/num_of_people, &current_gradient);
			/*
			Vector next_gradient;
					
			//cout<<"current_sample_size for norm calculation="<<current_sample_size<<endl;
			objective.ComputeGradient(current_sample_size, current_parameter, &next_gradient);
		//la::Scale(1.0/current_added_first_stage_
			//NOTIFY("current_parameter for the  calculation of norm");
			//for(index_t i=0; i<current_parameter.length(); i++){
			//	cout<<current_parameter[i]<<" ";
			//}
			//cout<<endl;
			//NOTIFY("Gradient for the calculation of norm");
			cout<<"Gradient for the calculation of norm"<<endl;
			for(index_t i=0; i<current_parameter.length(); i++){
				cout<<next_gradient[i]<<" ";
			}
			cout<<endl;
      */


			gradient_norm = sqrt(la::Dot(current_gradient, current_gradient));
			
			cout<<"gradient_norm="<<gradient_norm<<endl;

			if(gradient_norm<zero_tolerance){
				NOTIFY("Gradient norm is small enough...Exit...");
				cout<<"Gradient norm is small enough...Exit..."<<endl;
				break;
			}



			

			//NOTIFY("All data are used");
				}



		/*
    NOTIFY("True hessian");
    cout<<"True hessian"<<endl;
		Matrix opt_hessian;
		objective.ComputeHessian(current_sample_size, tpar, &opt_hessian);
		//la::Scale(1.0/current_added_first_stage_x.size(), &opt_hessian);
		
		cout<<"Hessian matrix at true par: "<<endl;

		
		for (index_t j=0; j<opt_hessian.n_rows(); j++){
			for (index_t k=0; k<opt_hessian.n_cols(); k++){
				cout<<opt_hessian.get(j,k) <<"  ";
			}
			cout<<endl;
		}
		*/

    
		/*
		///////////////////////////////////////////////////////////////////////
    ///////////////////////////////Exact hessian calculation
		//NOTIFY("Exact hessian calculation starts");

		Matrix current_hessian;
		objective.ComputeHessian(current_sample_size, current_parameter, &current_hessian);
		//la::Scale(1.0/current_added_first_stage_x.size(), &current_hessian);
		
		cout<<"Hessian matrix: "<<endl;
    //cout<<"approx_hessian"<<endl;
		for (index_t j=0; j<current_hessian.n_rows(); j++){
			for (index_t k=0; k<current_hessian.n_cols(); k++){
				cout<<current_hessian.get(j,k) <<"  ";
			}
			cout<<endl;
		}

*/

		
/*
		NOTIFY("Hessian approximation");
		cout<<"Hessian approximation"<<endl;
		Matrix approx_hessian;
		objective.CheckHessian(current_sample_size, current_parameter, &approx_hessian);

		cout<<"approx_hessian1"<<endl;
		for (index_t j=0; j<current_hessian.n_rows(); j++){
			for (index_t k=0; k<current_hessian.n_cols(); k++){
				cout<<approx_hessian.get(j,k) <<"  ";
			}
			cout<<endl;
		}

		Matrix approx_hessian2;
		objective.CheckHessian2(current_sample_size, current_parameter, &approx_hessian2);


		cout<<"approx_hessian2"<<endl;
		for (index_t j=0; j<current_hessian.n_rows(); j++){
			for (index_t k=0; k<current_hessian.n_cols(); k++){
				cout<<approx_hessian2.get(j,k) <<"  ";
			}
			cout<<endl;
		}
*/




/////////////////////////////////////////////////////////////////


/*
		for (index_t j=0; j<current_hessian.n_rows(); j++){
			for (index_t k=0; k<current_hessian.n_cols(); k++){
				cout<<current_hessian.get(j,k) <<"  ";
			}
			cout<<endl;
		}
		*/

		

		
		///////////////////////////////////////////////////////////////////////
    ///////////////////////////////hessian update - BFGS
		
		//if(iteration_count==1){

			//NOTIFY("Initial hessian matrix calculation...");

			//Matrix current_hessian;			
			//objective.ComputeHessian(current_sample_size, current_parameter, &current_hessian);
		
		//}		//iteration_count==1
		/*

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
		*/

  
	if(iteration_count==1){
		cout<<"iteration_count==1"<<endl;
		objective.CheckHessian3(current_sample_size, current_parameter, &current_hessian);
/*
		cout<<"hessian0"<<endl;
		for (index_t j=0; j<current_hessian.n_rows(); j++){
			for (index_t k=0; k<current_hessian.n_cols(); k++){
				cout<<current_hessian.get(j,k) <<"  ";
			}
			cout<<endl;
		}
		
*/
	}
	  //Matrix current_hessian;
		


    //NOTIFY("Dogleg method starts...");
		//cout<<"Current Hessian matrix"<<endl;
		//cout<<current_hessian.get(0,0)<<endl;

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
																					
																					
		//cout<<"test?="<<endl;			
    optimization.ComputeDerectionNoConstraints(current_radius, 
																					current_gradient,
																					current_hessian,
																					current_parameter,
																					&current_p,
																					&current_delta_m,
																					&next_parameter,
																					&new_radius);*/

		optimization.ComputeDerectionUnderAlphaConstraints(current_radius, 
																					current_gradient,
																					current_hessian,
																					current_parameter,
																					&current_p,
																					&current_delta_m,
																					&next_parameter,
																					&new_radius);
																					
																					
		//cout<<"where"<<endl;
		
		

    /*
		//Scaled version
		optimization.ComputeScaledDerectionUnderConstraints(current_radius, 
																					current_gradient,
																					current_hessian,
																					current_parameter,
																					&current_p,
																					&current_delta_m,
																					&next_parameter,
																					&new_radius);
																					*/


			

										
										


		 


		current_radius=new_radius;
		cout<<"current_radius="<<current_radius<<endl;

				
		p_norm=sqrt(la::Dot(current_p, current_p));
		 
		cout<<"candidate_new_parameter=";
		for(index_t i=0; i<next_parameter.length(); i++){
			cout<<next_parameter[i]<<" ";
		}
		cout<<endl;

		objective.ComputeObjective(current_sample_size, next_parameter, 
														 &next_objective);
		//next_objective/=current_added_first_stage_x.size();
		NOTIFY("The candidate Next objective is %g", next_objective);
		cout<<"The candidate Next objective is "<<next_objective<<endl;
	



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
			//double serror_factor=10*sample_size;
			//double serror_factor=0.055*sample_size;
			double serror_factor=0.02*sample_size;
			cout<<"serror_factor="<<serror_factor<<endl;
			//serror_factor=0.5;
			//cout<<"serror_factor="<<serror_factor<<endl;
			double sampling_deviation=0;

			for(index_t n=0; n<sample_size; n++){
				sampling_error+= pow(((current_choice_probability[n]-next_choice_probability[n])-(current_objective-next_objective) ),2);
			}
			sampling_error*=(correction_factor/(sample_size*(sample_size-1)));
			sampling_deviation=sqrt(sampling_error);

			cout<<"sampling_error="<<sampling_error<<endl;
			cout<<"sampling_deviation="<<sampling_deviation<<endl;
			cout<<"current_delta_m="<<current_delta_m<<endl;

			current_percent_added_sample=0;

			//if((-1.0*current_delta_m)<serror_factor*sampling_error){	//current_delta_m is negative in our case
      if((-1.0*current_delta_m)<serror_factor*sampling_deviation){	//current_delta_m is negative in our case
				NOTIFY("Expand sample size");
				cout<<"Expand sample size"<<endl;
				//break;
				//current_percent_added_sample=(serror_factor*sampling_error/current_delta_m)*(serror_factor*sampling_error/current_delta_m)-1.0;
				//current_percent_added_sample=(serror_factor*sampling_deviation/current_delta_m)*(serror_factor*sampling_deviation/current_delta_m)-1.0;
				current_percent_added_sample=(serror_factor*sampling_deviation/(-1*current_delta_m))-1.0;
				current_percent_added_sample*=100.0;
				cout<<"percent_added_sample="<<current_percent_added_sample<<endl;
				cout<<"Extended sample size="<<current_percent_added_sample/100.0*sample_size<<endl;
			}

		}	//if

				/*
		if(sample_size>=num_of_people &&end_sampling==0){
			end_sampling+=1;
			NOTIFY("All data are used");
			cout<<"All data are used"<<endl;
		}
		


		cout<<"next_parameter_test"<<endl;
		for(index_t i=0; i<next_parameter.length(); i++){
			cout<<next_parameter[i] <<" ";
		}
		cout<<endl;

		cout<<"current_sample_size_test="<<current_sample_size<<endl;
    */

		Vector next_gradient;
		objective.ComputeGradient(current_sample_size, next_parameter, &next_gradient);

		/*
		cout<<"next_gradient"<<endl;
		for(index_t i=0; i<next_gradient.length(); i++){
			cout<<next_gradient[i] <<" ";
		}
		cout<<endl;	
    */

		//agreement rho calculation
		cout<<"delta_m="<<current_delta_m<<endl;
		//rho=1.0*(current_objective-next_objective)/(current_delta_m)*current_added_first_stage_x.size();
		rho= -1.0*(current_objective-next_objective)/(current_delta_m);

		
			////////////////////////////////////////////////////////
/*
		//hessian update
		//BFGS Hnew=Hcurrent-(Hc ss' Hc)/s'Hcs + yy'/s'y
		//where s=theta_c-theta, y=grad(theta_c)-grad(theta)
	Matrix diff_gradient;
	diff_gradient.Init(num_of_parameter, 1);
	diff_gradient.SetZero();

	Matrix diff_par;
	diff_par.Init(num_of_parameter, 1);
	diff_par.SetZero();

	Matrix temp1; //H*s*s'*H
	temp1.Init(num_of_parameter, num_of_parameter);
	temp1.SetZero();

	Matrix Hs;
	Hs.Init(num_of_parameter, num_of_parameter);

	double scale_temp1;
	scale_temp1=0;

	Matrix temp2; //yy'/s'y
	temp2.Init(num_of_parameter, num_of_parameter);
	*/
		
		//Hessian update - BFGS
		
		//Matrix updated_hessian;
		//updated_hessian.Init(current_gradient.length(), current_gradient.length());
		//updated_hessian.SetZero();

		la::Scale(-1.0, &current_hessian);
		la::Scale(-1.0, &current_gradient);

		la::SubOverwrite(current_gradient, next_gradient, &diff_gradient);
		//la::SubOverwrite(current_parameter, next_parameter, &diff_par);
		la::SubOverwrite(next_parameter, current_parameter, &diff_par);

		//la::SubOverwrite(next_gradient, current_gradient, &diff_gradient);
		//la::SubOverwrite(next_parameter, current_parameter, &diff_par);


		//cout<<"test"<<endl;
		
    Matrix mtx_diff_par;
		mtx_diff_par.Alias(diff_par.ptr(), diff_par.length(), 1);

		Matrix mtx_diff_gradient;
		mtx_diff_gradient.Alias(diff_gradient.ptr(), diff_gradient.length(), 1);

		/*
    cout<<"diff_gradient"<<endl;
		for(index_t i=0; i<diff_gradient.length(); i++){
			cout<<mtx_diff_gradient.get(i,0) <<" ";
		}
		cout<<endl;
    */

		
		//Matrix temp1; //H*s*s'*H
		//Matrix Hs;
		la::MulOverwrite(current_hessian, mtx_diff_par, &Hs);
		
		la::MulTransBOverwrite(Hs, Hs, &temp1);
		
		//double scale_temp1;
		scale_temp1=la::Dot(Hs, mtx_diff_par);
		la::Scale( (1.0/scale_temp1), &temp1);

		
    //Matrix temp2; //yy'/s'y
		la::MulTransBOverwrite(mtx_diff_gradient, mtx_diff_gradient, &temp2);
		//cout<<"temp2"<<temp2[0]<<endl;

		/*
		cout<<"mtx_diff_gradient"<<endl;
		for(index_t i=0; i<mtx_diff_gradient.n_rows(); i++){
			cout<<mtx_diff_gradient.get(i,0) <<" ";
		}
		cout<<endl;
    

		cout<<"temp1"<<endl;
		for (index_t j=0; j<updated_hessian.n_rows(); j++){
			for (index_t k=0; k<updated_hessian.n_cols(); k++){
				cout<<temp1.get(j,k) <<"  ";
			}
			cout<<endl;
		}


		la::Scale( (1/la::Dot(mtx_diff_par, mtx_diff_gradient)), &temp2);


		cout<<"temp2"<<endl;
		for (index_t j=0; j<updated_hessian.n_rows(); j++){
			for (index_t k=0; k<updated_hessian.n_cols(); k++){
				cout<<temp2.get(j,k) <<"  ";
			}
			cout<<endl;
		}
		*/

		//cout<<"dot"<<la::dot(mtx_diff_par, mtx_diff_gradient)<<endl;
		//cout<<"temp2"<<temp2[0]<<endl;

		la::SubOverwrite(temp1, current_hessian, &updated_hessian);
		la::AddTo(temp2, &updated_hessian);
		
		la::Scale(-1.0, &current_gradient);
		la::Scale(-1.0, &updated_hessian);
		la::Scale(-1.0, &current_hessian);
		cout<<"Hessian update"<<endl;
		
		/*
		cout<<"current_hessian"<<endl;
		for (index_t j=0; j<updated_hessian.n_rows(); j++){
			for (index_t k=0; k<updated_hessian.n_cols(); k++){
				cout<<current_hessian.get(j,k) <<"  ";
			}
			cout<<endl;
		}

		cout<<"update_hessian"<<endl;
		for (index_t j=0; j<updated_hessian.n_rows(); j++){
			for (index_t k=0; k<updated_hessian.n_cols(); k++){
				cout<<updated_hessian.get(j,k) <<"  ";
			}
			cout<<endl;
		}
		*/


		//Check positive definiteness
		/*
		Vector eigen_hessian;
		la::EigenvaluesInit(updated_hessian, &eigen_hessian);

		cout<<"eigen values of updated hessian"<<endl;

		for(index_t i=0; i<eigen_hessian.length(); i++){
			cout<<eigen_hessian[i]<<" ";
		}
		cout<<endl;
		*/
		
		Vector eigen_hessian;
		Matrix eigenvec_hessian;

    
		la::EigenvectorsInit(updated_hessian, &eigen_hessian, &eigenvec_hessian);

		cout<<"eigen values of updated hessian"<<endl;

		for(index_t i=0; i<eigen_hessian.length(); i++){
			cout<<eigen_hessian[i]<<" ";
		}
		cout<<endl;
		cout<<endl;

/*
		cout<<"eigen vectors of updated hessian"<<endl;
    		for (index_t j=0; j<updated_hessian.n_rows(); j++){
			for (index_t k=0; k<updated_hessian.n_cols(); k++){
				cout<<eigenvec_hessian.get(j,k) <<"  ";
			}
			cout<<endl;
		}
		cout<<endl;
*/

		/*
		double max_eigen=0;
		//cout<<"eigen_value:"<<endl;
		for(index_t i=0; i<eigen_hessian.length(); i++){
			//cout<<eigen_hessian[i]<<" ";
			if(eigen_hessian[i]>max_eigen){
				max_eigen=eigen_hessian[i];
			}

		}
		//cout<<endl;
		cout<<"max_eigen="<<(max_eigen)<<endl;
		*/



    


		
		cout<<"rho= "<<rho<<endl;
		if(rho>eta){
			current_parameter.CopyValues(next_parameter);
			NOTIFY("Accepting the step...");
			cout<<"Accepting the step"<<endl;

			cout<<"diff="<<(current_objective-next_objective)<<endl;
			//cout<<"current_radius="<<current_radius<<endl;
			if(current_sample_size>=num_of_people && iteration_count>1){
				if(current_objective-next_objective < 0.001 && current_radius <0.01){
					cout<<"Improvement in objective fn is small enough...Exit..."<<(current_objective-next_objective)<<endl;
					cout<<"current radius is small enough...Exit..."<<current_radius<<endl;
					break;
				}
			}


			/*
			if(current_radius <0.001) {
				cout<<"current radius is small enough...Exit..."<<current_radius<<endl;
				break;
			}
			*/

			current_hessian.CopyValues(updated_hessian);
			
			NOTIFY("Update the hessian matrix by BFGS method...");
			cout<<"Update the hessian matrix by BFGS method..."<<endl;		

		}
		
		//radius update
		optimization.TrustRadiusUpdate(rho, p_norm, &current_radius);

		



	}
	
		



	
	cout<<"Total_iteration_count="<<iteration_count<<endl;
	NOTIFY("Final solution: ");
	cout<<"Final solution"<<endl;
	for(index_t i=0; i<current_parameter.length(); i++) {
		cout<<current_parameter[i]<<" ";
	}
	cout<<endl;
  
	//cout<<"num_of_people="<<num_of_people;

	
	
	//Compute Variance of the estimates -H^{-1}
	Matrix final_hessian1;
	//objective.ComputeHessian(current_added_first_stage_x.size(), current_parameter, &final_hessian);
	
	final_hessian1.Alias(current_hessian);

	cout<<"Final_hessian1"<<endl;
	for (index_t j=0; j<final_hessian1.n_rows(); j++){
		for (index_t k=0; k<final_hessian1.n_cols(); k++){
			cout<<final_hessian1.get(j,k) <<"  ";
		}
		cout<<endl;
	}

	Matrix inverse_hessian1;
	if( !PASSED(la::InverseInit(final_hessian1, &inverse_hessian1)) ) {
		NOTIFY("Final hessian1 matrix is not invertible!");
	}
	else{
		la::Scale(-1.0, &inverse_hessian1);

		/*
		cout<<"Diagonal of inverse final hessian: ";
		for(index_t i=0; i<inverse_hessian1.n_rows(); i++){
			cout<<inverse_hessian1.get(i,i)<<" ";
		}
		cout<<endl;
		*/
		
		index_t n=inverse_hessian1.n_rows();
		Vector estimates_variance1;
		estimates_variance1.Init(n);
		for(index_t i=0; i<n; i++) {
			estimates_variance1[i]=inverse_hessian1.get(i,i);
		}


		//linalg__private::DiagToVector(inverse_hessian, &estimates_variance);
		
		cout<<"Variance of etimates: ";
		for(index_t i=0; i<estimates_variance1.length(); i++) {
			cout<<estimates_variance1[i]<<" ";

		}
		cout<<endl;

	}
  
	NOTIFY("Final hessian calculation2");
	/*
	cout<<"Final hessian calculation2"<<endl;
	Matrix final_hessian;
	objective.CheckHessian(num_of_people, current_parameter, &final_hessian);

	cout<<"Final_hessian2 from finite approx."<<endl;
	for (index_t j=0; j<final_hessian.n_rows(); j++){
		for (index_t k=0; k<final_hessian.n_cols(); k++){
			cout<<final_hessian.get(j,k) <<"  ";
		}
		cout<<endl;
	}
  	cout<<endl;
	cout<<endl;

	Matrix eigenvec_hessian2;
	Vector eigen_hessian2;
	la::EigenvectorsInit(final_hessian, &eigen_hessian2, 
																					&eigenvec_hessian2);

	cout<<"eigen values of final hessian-finite-approx"<<endl;

	for(index_t i=0; i<eigen_hessian2.length(); i++){
		cout<<eigen_hessian2[i]<<" ";
	}
	cout<<endl;
	cout<<endl;

	cout<<"eigen vectors of final hessian-finite-approx"<<endl;
  	for (index_t j=0; j<final_hessian.n_rows(); j++){
		for (index_t k=0; k<final_hessian.n_cols(); k++){
			cout<<eigenvec_hessian2.get(j,k) <<"  ";
		}
		cout<<endl;
	}
	cout<<endl;

	Matrix inverse_hessian;
	if( !PASSED(la::InverseInit(final_hessian, &inverse_hessian)) ) {
		NOTIFY("Final hessian matrix is not invertible!");
	}
	else{
		la::Scale(-1.0, &inverse_hessian);

		cout<<"Diagonal of inverse final hessian2: ";
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
*/

	/*

	//NOTIFY("Robust covariance matrix calculation");

	current_gradient
	*/


	


	

	

 
  fx_done(module);
}













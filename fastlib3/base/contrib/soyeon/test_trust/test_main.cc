#include "fastlib/fastlib.h"
#include "test_obj.h"
#include "optimization.h"
#include "iostream.h"


int main(int argc, char *argv[]) {


  fx_module *fx_root = fx_init(argc, argv, NULL);


  RosenbrockFunction my_test;
	Optimization optimization;

  //optional input file, default file "xinit.csv"
  const char* initial_solution_file = fx_param_str(fx_root, "xinit_file", "xinit.csv");
  
  /*
	//Reguired input file
  const char* quadratic_term_file = fx_param_str_req(fx_root, "qdata");
  const char* linear_term_file = fx_param_str_req(fx_root, "ldata");
  //index_t problem_id = fx_param_int_req(fx_root, "problem_id");

  Vector initial_x;
  Matrix quadratic_term;
  Vector linear_term;
  */

  //fastlib only load data as Matrix
	Vector initial_x;
  Matrix initial_x_mat;
  if (data::Load(initial_solution_file, &initial_x_mat)==SUCCESS_FAIL) {
    FATAL("File %s not found", initial_solution_file);
  };  
  initial_x_mat.MakeColumnVector(0, &initial_x);
  
	/*
	if (data::Load(quadratic_term_file, &quadratic_term)==SUCCESS_FAIL) {
    FATAL("File %s not found", quadratic_term_file);
  } 
  Matrix linear_term_mat;
  data::Load(linear_term_file, &linear_term_mat);
  linear_term_mat.MakeColumnVector(0, &linear_term);
	*/

  //my_test.Init(quadratic_term, linear_term);

	int num_of_parameter=initial_x.length();
	Vector current_parameter;
	//current_parameter.Init(num_of_parameter);
	current_parameter.Copy(initial_x);
	cout<<"Initial points="<<endl;
	for(index_t i=0; i<initial_x.length(); i++){
		cout<<current_parameter[i]<<" ";
	}
	cout<<endl;
	Vector next_parameter;
	//cout<<"num par="<<num_of_parameter<<endl;
	next_parameter.Init(num_of_parameter);
	next_parameter.SetZero();

	double current_objective;
	double next_objective;
	double p_norm=0;

	double current_radius=0.001;	//initial_radius
	double eta=0.2;
	
	double rho=0; //agreement(ratio)

	double error_tolerance=1e-16;
	double zero_tolerance=1e-2;	//for gradient norm

	int max_iteration=2000;
	int iteration_count=0;

	while(iteration_count<max_iteration){

		
		iteration_count+=1;
		cout<<"iteration_count="<<iteration_count<<endl;

		my_test.ComputeObjective(current_parameter, 
															 &current_objective);
		NOTIFY("The objective is %g", current_objective);

		
		//NOTIFY("Gradient calculation starts");
		Vector current_gradient;
		//gradient.Init(num_of_betas_);
		my_test.ComputeGradient(current_parameter, &current_gradient);
		cout<<"gradient="<<endl;
		for(index_t i=0; i<current_gradient.length(); i++){
			cout<<current_gradient[i]<<" ";
		}
		cout<<endl;


		Matrix current_hessian;
		my_test.ComputeHessian(current_parameter, &current_hessian);
		
		/*
		cout<<"Hessian matrix: "<<endl;
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

		p_norm=sqrt(la::Dot(current_p, current_p));
		 
		cout<<"new_parameter=";
		for(index_t i=0; i<next_parameter.length(); i++){
			cout<<next_parameter[i]<<" ";
		}
		cout<<endl;

		my_test.ComputeObjective(next_parameter, 
														 &next_objective);
		NOTIFY("The Next objective is %g", next_objective);

		//agreement rho calculation
		rho=+1.0*(current_objective-next_objective)/(current_delta_m);
		cout<<"rho= "<<rho<<endl;
		if(rho>eta){
			current_parameter.CopyValues(next_parameter);
			NOTIFY("Accepting the step...");
		}
		
		//radius update
		optimization.TrustRadiusUpdate(rho, p_norm, &current_radius);

		double gradient_norm;
		//la::Scale(1.0/num_of_people, &current_gradient);
		gradient_norm = sqrt(la::Dot(current_gradient, current_gradient));
		cout<<"gradient_norm="<<gradient_norm<<endl;

		if(gradient_norm<zero_tolerance){
			NOTIFY("Gradient norm is small enough...Exit...");
			break;
		}


	}	//while
	
		

	cout<<"Total_iteration_count="<<iteration_count<<endl;
	NOTIFY("Final solution: ");
	for(index_t i=0; i<current_parameter.length(); i++) {
		cout<<current_parameter[i]<<" ";
	}
	cout<<endl;

	//Compute Variance of the estimates -H^{-1}
	Matrix final_hessian;
	my_test.ComputeHessian(current_parameter, &final_hessian);
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
	}	//else
	

  
	

  fx_done(fx_root);





}



  

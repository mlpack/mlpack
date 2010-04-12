#include "Dsampling.h"
#include <stdlib.h>	//rand() srand()
#include <time.h>	//time()
#include <algorithm>	//swap
#include <iostream>
using namespace std;


void DSampling::Init(fx_module *module, int *num_of_people, 
											Vector *ind_unknown_x, double *initial_percent_sample,
											Vector *initial_parameter) {
	module_=module;
	const char *data_file1=fx_param_str_req(module_, "data1");
	const char *info_file1=fx_param_str_req(module_, "info1");
	Matrix x;
	data::Load(data_file1, &x);
	index_t num_of_betas=x.n_rows();
	
  Matrix info1;
  data::Load(info_file1, &info1);
	num_of_people_=info1.n_cols();
	
	//Experiment with 5000 ppl
	//num_of_people_=5000;
	
	*num_of_people=num_of_people_;
  population_first_stage_x_.Init(num_of_people_);
	index_t start_col=0;
  for(index_t i=0; i<num_of_people_; i++) {
    population_first_stage_x_[i].Init(x.n_rows(), (index_t)info1.get(0, i));
    population_first_stage_x_[i].CopyColumnFromMat(0, start_col, 
										(index_t)info1.get(0,i), x);
    start_col+=(index_t)info1.get(0, i);
  }

	/*
	cout<<"data file 1:";
	for(index_t i=0; i<num_of_people_; i++) {
		cout<<"population_first_stage_x["<<i<<"]"<<endl;
		for(index_t j=0; j<population_first_stage_x_[i].n_rows(); j++){
			for(index_t k=0; k<population_first_stage_x_[i].n_cols(); k++) {
				cout<<population_first_stage_x_[i].get(j,k)<<" ";
			}
			cout<<endl;
		}
		cout<<endl;		
	}
	*/
  const char *data_file2=fx_param_str_req(module_, "data2");
	x.Destruct();
	data::Load(data_file2, &x);
	population_second_stage_x_.Init(num_of_people_);
	start_col=0;
  for(index_t i=0; i<num_of_people_; i++) {
    population_second_stage_x_[i].Init(x.n_rows(), (index_t)info1.get(0,i));
	  population_second_stage_x_[i].CopyColumnFromMat(0, start_col, 
							(index_t)info1.get(0,i), x);
    start_col+=(index_t)info1.get(0,i);
  }

	/*cout<<"data file 2:";
	for(index_t i=0; i<num_of_people_; i++) {
		cout<<"population_second_stage_x["<<i<<"]"<<endl;
		for(index_t j=0; j<population_second_stage_x_[i].n_rows(); j++){
			for(index_t k=0; k<population_second_stage_x_[i].n_cols(); k++) {
				cout<<population_second_stage_x_[i].get(j,k)<<" ";
			}
			cout<<endl;
		}
		cout<<endl;		
	}
	*/

	//need to be fixed
	const char *data_file3=fx_param_str_req(module_, "data3");
  x.Destruct();
  data::Load(data_file3, &x);
  population_unknown_x_past_.Init(num_of_people_);
  start_col=0;
  for(index_t i=0; i<num_of_people_; i++) {
    population_unknown_x_past_[i].Init(x.n_rows(), (index_t)info1.get(0,i));
    population_unknown_x_past_[i].CopyColumnFromMat(0, start_col, 
        (index_t)info1.get(0,i), x);
    start_col+=(index_t)info1.get(0, i);
  }

	 
	const char *info_y_file=fx_param_str_req(module_, "info_y");
  Matrix info_y;
  data::Load(info_y_file, &info_y);
  population_first_stage_y_.Init(num_of_people_);
  for(index_t i=0; i<num_of_people_; i++) {
	  population_first_stage_y_[i]=(index_t)info_y.get(0,i);
  }

  const char *info_ind_unknown_x_file=fx_param_str_req(module_, "info_ind_unknown_x");
  Matrix info_ind_unknown_x;
  data::Load(info_ind_unknown_x_file, &info_ind_unknown_x);
  num_of_unknown_x_=info_ind_unknown_x.n_cols();
  population_ind_unknown_x_.Init(num_of_unknown_x_);
  for(index_t i=0; i<num_of_unknown_x_; i++) {
	  population_ind_unknown_x_[i]=info_ind_unknown_x.get(0,i);
  }

	ind_unknown_x->Copy(population_ind_unknown_x_);

	
	if (fx_param_exists(module_, "starting_points")) {
		const char *initial_parameter_file=fx_param_str_req(module_,"starting_points");
		Matrix mtx_initial_parameter;
		if(data::Load(initial_parameter_file, &mtx_initial_parameter)==SUCCESS_FAIL) {
			FATAL("File %s not found", initial_parameter_file);
		}
		else{

			if(mtx_initial_parameter.n_rows() != num_of_betas+1){
				NOTIFY("The number of starting points given is not same as the number of parameters");
				NOTIFY("Use default...");
				//initial_parameter->Destruct(); 
				initial_parameter->Init(num_of_betas+1);	//beta+alpha
				//initial_parameter->SetZero();
				//(*initial_parameter)[num_of_betas]=2;
				//(*initial_parameter)[num_of_betas+1]=2;
				initial_parameter->SetAll(0.5);

				
				
			}	//if
			else{
				mtx_initial_parameter.MakeColumnVector(0, initial_parameter);
				/*
				cout<<"Starting points are:   ";
				for(index_t i=0; i<initial_parameter->length(); i++){
					cout<<(*initial_parameter)[i]<<" ";
				}
				cout<<endl;
				*/
			}
		}	//else
	}	//if
	else {
		NOTIFY("Starting points are not given. Use default...");
		//NOTIFY("Number of parameters is %d", num_of_betas+2);
		cout<<"Number of parameters is "<<(num_of_betas+1)<<endl;
		initial_parameter->Init(num_of_betas+1);
		
		//initial_parameter->SetAll(1.5);

		
		initial_parameter->SetAll(0.5);	//0<alpha<=1
		
	  cout<<"NUM_OF_BETAS="<<num_of_betas<<endl;
		//(*initial_parameter)[num_of_betas]=2;
		//(*initial_parameter)[num_of_betas+1]=2;
		
/*			
		(*initial_parameter)[0]=1.2;
		(*initial_parameter)[1]=1.5;
		(*initial_parameter)[num_of_betas]=1.7;
		(*initial_parameter)[num_of_betas+1]=3;
		
			
		
		(*initial_parameter)[0]=1.3392;
		(*initial_parameter)[1]=1.62639;
		(*initial_parameter)[num_of_betas]=2.00565;
		(*initial_parameter)[num_of_betas+1]=1.32168;
		


		(*initial_parameter)[0]=1.34071;
		(*initial_parameter)[1]=1.6346;
		(*initial_parameter)[num_of_betas]=1.67677;
		(*initial_parameter)[num_of_betas+1]=1.09795;

		


		(*initial_parameter)[0]=4.89444;
		(*initial_parameter)[1]=-1.36876;
		(*initial_parameter)[num_of_betas]=0.267926;
		(*initial_parameter)[num_of_betas+1]=1.64124e-12;*/

		      		
		



		/*
		cout<<"Starting points are:   ";
		for(index_t i=0; i<initial_parameter->length(); i++){
			cout<<(*initial_parameter)[i]<<" ";
		}
		cout<<endl;
		*/

		//initial_parameter[num_or_betas]->2;
		//initial_parameter[num_or_betas+1]->2;


	}



	//const char *initial_parameter_file=fx_param_str(module_, "stating_points", "default_initial.csv");
	/*if(initial_parameter_file !=NULL) {
		Matrix mtx_initial_parameter;
    data::Load(initial_parameter_file, &mtx_initial_parameter);
		mtx_initial_parameter.MakeColumnVector(0, initial_parameter);
	}
	else {
		NOTIFY("Starting points are not given. Use default...");
		initial_parameter->Init(x.n_rows());
		initial_parameter->SetZero();
	}
	*/
	/*Matrix mtx_initial_parameter;
  data::Load(initial_parameter_file, &mtx_initial_parameter);
	if(mtx_initial_parameter.n_rows()==1 && mtx_initial_parameter.get(0,0)==0){
		NOTIFY("Starting points are not given. Use default...");
		initial_parameter->Init(x.n_rows());
		initial_parameter->SetZero();
	}
	else if(mtx_initial_parameter.n_rows() != x.n_rows()){
		FATAL("The number of starting points given is not same as the number of parameters");
	}
	else {
		mtx_initial_parameter.MakeColumnVector(0, initial_parameter);
	}
	*/


  double arg_initial_percent_sample=fx_param_double(module_, "initial_percent_sample", num_of_people_*0.05);

	//Initilize memeber variables
	num_of_selected_sample_=0;
	count_num_sampling_=0;
	initial_percent_sample_=arg_initial_percent_sample;
  *initial_percent_sample=initial_percent_sample_;

	shuffled_array_.Init(num_of_people_);

	//return num_of_people_;

}

void DSampling::Init2(fx_module *module, int *num_of_people, 
											Vector *ind_unknown_x, double *initial_percent_sample,
											Vector *initial_parameter) {
	module_=module;
	const char *data_file1=fx_param_str_req(module_, "data1");
	const char *info_file1=fx_param_str_req(module_, "info1");
	Matrix x;
	data::Load(data_file1, &x);

/*
	//Check condition number of data matrix without intercept
	Matrix cx;
	la::MulTransBInit(x, x, &cx);

	Vector eigen_cx;
	la::EigenvaluesInit(cx, &eigen_cx);

	for(index_t i=0; i<eigen_cx.length(); i++){
		cout<<eigen_cx[i]<<" ";
	}
	cout<<endl;

	double max_eigen_cx=eigen_cx[0];
	double min_eigen_cx=eigen_cx[0];
	
	for(index_t i=0; i<eigen_cx.length(); i++){
		if(eigen_cx[i]>max_eigen_cx){
			max_eigen_cx=eigen_cx[i];
		}
		else if(eigen_cx[i]<min_eigen_cx){
			min_eigen_cx=eigen_cx[i];
		}
	}

	cout<<"without intercept"<<endl;
	cout<<"min_eigen_cx="<<min_eigen_cx<<endl;
	cout<<"max_eigen_cx="<<max_eigen_cx<<endl;
	cout<<"radio eigen="<<max_eigen_cx/min_eigen_cx<<endl;

*/


	Vector intercept;
	intercept.Init(x.n_cols());
	intercept.SetAll(1);
	
	Matrix x2;
	la::TransposeInit(x, &x2); 
	  		
	Matrix x3;
	x3.Init(x.n_cols(), x.n_rows()+1);
	x3.SetAll(1);
	x3.CopyColumnFromMat(1, 0, x.n_rows(), x2);
	
	x.Destruct();
	la::TransposeInit(x3, &x); 
	index_t num_of_betas=x.n_rows();

	//check condition number WITH intercept
	Matrix cx;
	la::MulTransBInit(x, x, &cx);

	Vector eigen_cx;
	la::EigenvaluesInit(cx, &eigen_cx);

	for(index_t i=0; i<eigen_cx.length(); i++){
		cout<<eigen_cx[i]<<" ";
	}
	cout<<endl;

	double max_eigen_cx=eigen_cx[0];
	double min_eigen_cx=eigen_cx[0];
	
	for(index_t i=0; i<eigen_cx.length(); i++){
		if(eigen_cx[i]>max_eigen_cx){
			max_eigen_cx=eigen_cx[i];
		}
		else if(eigen_cx[i]<min_eigen_cx){
			min_eigen_cx=eigen_cx[i];
		}
	}
	cout<<"condition number WITH intercept"<<endl;
	cout<<"min_eigen_cx="<<min_eigen_cx<<endl;
	cout<<"max_eigen_cx="<<max_eigen_cx<<endl;
	cout<<"ratio eigen="<<max_eigen_cx/min_eigen_cx<<endl;

	
	
	Matrix info1;
       data::Load(info_file1, &info1);
	num_of_people_=info1.n_cols();
	
	
  
	//Experiment with 5000 ppl
	//num_of_people_=5000;
	
	*num_of_people=num_of_people_;
  population_first_stage_x_.Init(num_of_people_);
	index_t start_col=0;
  for(index_t i=0; i<num_of_people_; i++) {
    population_first_stage_x_[i].Init(x.n_rows(), (index_t)info1.get(0, i));
    population_first_stage_x_[i].CopyColumnFromMat(0, start_col, 
										(index_t)info1.get(0,i), x);
    start_col+=(index_t)info1.get(0, i);
  }

	/*
	cout<<"data file 1:";
	for(index_t i=0; i<num_of_people_; i++) {
		cout<<"population_first_stage_x["<<i<<"]"<<endl;
		for(index_t j=0; j<population_first_stage_x_[i].n_rows(); j++){
			for(index_t k=0; k<population_first_stage_x_[i].n_cols(); k++) {
				cout<<population_first_stage_x_[i].get(j,k)<<" ";
			}
			cout<<endl;
		}
		cout<<endl;		
	}
	*/
  const char *data_file2=fx_param_str_req(module_, "data2");
	x.Destruct();
	data::Load(data_file2, &x);

	x2.Destruct();
  	la::TransposeInit(x, &x2);   
       
	
	x3.Destruct();
	x3.Init(x.n_cols(), x.n_rows()+1);
	x3.SetAll(1);
	x3.CopyColumnFromMat(1, 0, x.n_rows(), x2);
	

	x.Destruct();
	la::TransposeInit(x3, &x); 

	population_second_stage_x_.Init(num_of_people_);
	start_col=0;
  for(index_t i=0; i<num_of_people_; i++) {
    population_second_stage_x_[i].Init(x.n_rows(), (index_t)info1.get(0,i));
	  population_second_stage_x_[i].CopyColumnFromMat(0, start_col, 
							(index_t)info1.get(0,i), x);
    start_col+=(index_t)info1.get(0,i);
  }

	/*cout<<"data file 2:";
	for(index_t i=0; i<num_of_people_; i++) {
		cout<<"population_second_stage_x["<<i<<"]"<<endl;
		for(index_t j=0; j<population_second_stage_x_[i].n_rows(); j++){
			for(index_t k=0; k<population_second_stage_x_[i].n_cols(); k++) {
				cout<<population_second_stage_x_[i].get(j,k)<<" ";
			}
			cout<<endl;
		}
		cout<<endl;		
	}
	*/

	//need to be fixed
	const char *data_file3=fx_param_str_req(module_, "data3");
  x.Destruct();
  data::Load(data_file3, &x);
  population_unknown_x_past_.Init(num_of_people_);
  start_col=0;
  for(index_t i=0; i<num_of_people_; i++) {
    population_unknown_x_past_[i].Init(x.n_rows(), (index_t)info1.get(0,i));
    population_unknown_x_past_[i].CopyColumnFromMat(0, start_col, 
        (index_t)info1.get(0,i), x);
    start_col+=(index_t)info1.get(0, i);
  }

	 
	const char *info_y_file=fx_param_str_req(module_, "info_y");
  Matrix info_y;
  data::Load(info_y_file, &info_y);
  population_first_stage_y_.Init(num_of_people_);
  for(index_t i=0; i<num_of_people_; i++) {
	  population_first_stage_y_[i]=(index_t)info_y.get(0,i);
  }

  const char *info_ind_unknown_x_file=fx_param_str_req(module_, "info_ind_unknown_x");
  Matrix info_ind_unknown_x;
  data::Load(info_ind_unknown_x_file, &info_ind_unknown_x);
  num_of_unknown_x_=info_ind_unknown_x.n_cols();
  population_ind_unknown_x_.Init(num_of_unknown_x_);
  for(index_t i=0; i<num_of_unknown_x_; i++) {
	  population_ind_unknown_x_[i]=info_ind_unknown_x.get(0,i);
  }

	cout<<"ind_unknown_x="<<population_ind_unknown_x_[0]<<endl;
	ind_unknown_x->Copy(population_ind_unknown_x_);

	
	if (fx_param_exists(module_, "starting_points")) {
		const char *initial_parameter_file=fx_param_str_req(module_,"starting_points");
		Matrix mtx_initial_parameter;
		if(data::Load(initial_parameter_file, &mtx_initial_parameter)==SUCCESS_FAIL) {
			FATAL("File %s not found", initial_parameter_file);
		}
		else{

			if(mtx_initial_parameter.n_rows() != num_of_betas+1){
				NOTIFY("The number of starting points given is not same as the number of parameters");
				NOTIFY("Use default...");
				//initial_parameter->Destruct(); 
				initial_parameter->Init(num_of_betas+1);	//beta+p+q
				//initial_parameter->SetZero();
				//(*initial_parameter)[num_of_betas]=2;
				//(*initial_parameter)[num_of_betas+1]=2;
				initial_parameter->SetAll(0.5);

				
				
			}	//if
			else{
				mtx_initial_parameter.MakeColumnVector(0, initial_parameter);
				/*
				cout<<"Starting points are:   ";
				for(index_t i=0; i<initial_parameter->length(); i++){
					cout<<(*initial_parameter)[i]<<" ";
				}
				cout<<endl;
				*/
			}
		}	//else
	}	//if
	else {
		NOTIFY("Starting points are not given. Use default...");
		//NOTIFY("Number of parameters is %d", num_of_betas+2);
		cout<<"Number of parameters is "<<(num_of_betas+1)<<endl;
		initial_parameter->Init(num_of_betas+1);
		
		//initial_parameter->SetAll(1.5);

		
		initial_parameter->SetAll(0);
		(*initial_parameter)[num_of_betas]=0.5;
		
	cout<<"NUM_OF_BETAS="<<num_of_betas<<endl;
		//(*initial_parameter)[num_of_betas]=2;
		//(*initial_parameter)[num_of_betas+1]=2;
		
/*			
		(*initial_parameter)[0]=1.2;
		(*initial_parameter)[1]=1.5;
		(*initial_parameter)[num_of_betas]=1.7;
		(*initial_parameter)[num_of_betas+1]=3;
		
			
		
	



		(*initial_parameter)[0]=4.89444;
		(*initial_parameter)[1]=-1.36876;
		(*initial_parameter)[num_of_betas]=0.267926;
		(*initial_parameter)[num_of_betas+1]=1.64124e-12;*/

		      		
		



		/*
		cout<<"Starting points are:   ";
		for(index_t i=0; i<initial_parameter->length(); i++){
			cout<<(*initial_parameter)[i]<<" ";
		}
		cout<<endl;
		*/

		//initial_parameter[num_or_betas]->2;
		//initial_parameter[num_or_betas+1]->2;


	}



	//const char *initial_parameter_file=fx_param_str(module_, "stating_points", "default_initial.csv");
	/*if(initial_parameter_file !=NULL) {
		Matrix mtx_initial_parameter;
    data::Load(initial_parameter_file, &mtx_initial_parameter);
		mtx_initial_parameter.MakeColumnVector(0, initial_parameter);
	}
	else {
		NOTIFY("Starting points are not given. Use default...");
		initial_parameter->Init(x.n_rows());
		initial_parameter->SetZero();
	}
	*/
	/*Matrix mtx_initial_parameter;
  data::Load(initial_parameter_file, &mtx_initial_parameter);
	if(mtx_initial_parameter.n_rows()==1 && mtx_initial_parameter.get(0,0)==0){
		NOTIFY("Starting points are not given. Use default...");
		initial_parameter->Init(x.n_rows());
		initial_parameter->SetZero();
	}
	else if(mtx_initial_parameter.n_rows() != x.n_rows()){
		FATAL("The number of starting points given is not same as the number of parameters");
	}
	else {
		mtx_initial_parameter.MakeColumnVector(0, initial_parameter);
	}
	*/


  double arg_initial_percent_sample=fx_param_double(module_, "initial_percent_sample", num_of_people_*0.05);

	//Initilize memeber variables
	num_of_selected_sample_=0;
	count_num_sampling_=0;
	initial_percent_sample_=arg_initial_percent_sample;
  *initial_percent_sample=initial_percent_sample_;

	shuffled_array_.Init(num_of_people_);

	//return num_of_people_;

}


void DSampling::Init3(fx_module *module, int *num_of_people, 
											Vector *ind_unknown_x, double *initial_percent_sample,
											Vector *initial_parameter, Vector *opt_x) {
	module_=module;
	const char *data_file1=fx_param_str_req(module_, "data1");
	const char *info_file1=fx_param_str_req(module_, "info1");
	Matrix x;
	data::Load(data_file1, &x);

	/*
	//Check condition number of data matrix without intercept
	Matrix cx;
	la::MulTransBInit(x, x, &cx);

	Vector eigen_cx;
	la::EigenvaluesInit(cx, &eigen_cx);

	for(index_t i=0; i<eigen_cx.length(); i++){
		cout<<eigen_cx[i]<<" ";
	}
	cout<<endl;

	double max_eigen_cx=eigen_cx[0];
	double min_eigen_cx=eigen_cx[0];
	
	for(index_t i=0; i<eigen_cx.length(); i++){
		if(eigen_cx[i]>max_eigen_cx){
			max_eigen_cx=eigen_cx[i];
		}
		else if(eigen_cx[i]<min_eigen_cx){
			min_eigen_cx=eigen_cx[i];
		}
	}
	cout<<"condition number without intercept"<<endl;
	cout<<"min_eigen_cx="<<min_eigen_cx<<endl;
	cout<<"max_eigen_cx="<<max_eigen_cx<<endl;
	cout<<"ratio eigen="<<max_eigen_cx/min_eigen_cx<<endl;

	*/


	Vector intercept;
	intercept.Init(x.n_cols());
	intercept.SetAll(1);
	
	Matrix x2;
	la::TransposeInit(x, &x2); 
	  		
	Matrix x3;
	x3.Init(x.n_cols(), x.n_rows()+1);
	x3.SetAll(1);
	x3.CopyColumnFromMat(1, 0, x.n_rows(), x2);
	
	x.Destruct();
	la::TransposeInit(x3, &x); 
	index_t num_of_betas=x.n_rows();


	//check condition number WITH intercept
	Matrix cx;
	la::MulTransBInit(x, x, &cx);

	Vector eigen_cx;
	la::EigenvaluesInit(cx, &eigen_cx);

	for(index_t i=0; i<eigen_cx.length(); i++){
		cout<<eigen_cx[i]<<" ";
	}
	cout<<endl;

	double max_eigen_cx=eigen_cx[0];
	double min_eigen_cx=eigen_cx[0];
	
	for(index_t i=0; i<eigen_cx.length(); i++){
		if(eigen_cx[i]>max_eigen_cx){
			max_eigen_cx=eigen_cx[i];
		}
		else if(eigen_cx[i]<min_eigen_cx){
			min_eigen_cx=eigen_cx[i];
		}
	}
	cout<<"condition number WITH intercept"<<endl;
	cout<<"min_eigen_cx="<<min_eigen_cx<<endl;
	cout<<"max_eigen_cx="<<max_eigen_cx<<endl;
	cout<<"ratio eigen="<<max_eigen_cx/min_eigen_cx<<endl;

	
	Matrix info1;
    data::Load(info_file1, &info1);
	num_of_people_=info1.n_cols();
	
	
  
	//Experiment with 5000 ppl
	//num_of_people_=5000;
	
	*num_of_people=num_of_people_;
    population_first_stage_x_.Init(num_of_people_);
	index_t start_col=0;
    for(index_t i=0; i<num_of_people_; i++) {
      population_first_stage_x_[i].Init(x.n_rows(), (index_t)info1.get(0, i));
      population_first_stage_x_[i].CopyColumnFromMat(0, start_col, 
										(index_t)info1.get(0,i), x);
      start_col+=(index_t)info1.get(0, i);
    }

	/*
	cout<<"data file 1:";
	for(index_t i=0; i<num_of_people_; i++) {
		cout<<"population_first_stage_x["<<i<<"]"<<endl;
		for(index_t j=0; j<population_first_stage_x_[i].n_rows(); j++){
			for(index_t k=0; k<population_first_stage_x_[i].n_cols(); k++) {
				cout<<population_first_stage_x_[i].get(j,k)<<" ";
			}
			cout<<endl;
		}
		cout<<endl;		
	}
	*/
  const char *data_file2=fx_param_str_req(module_, "data2");
	x.Destruct();
	data::Load(data_file2, &x);

	x2.Destruct();
  	la::TransposeInit(x, &x2);   
       
	
	x3.Destruct();
	x3.Init(x.n_cols(), x.n_rows()+1);
	x3.SetAll(1);
	x3.CopyColumnFromMat(1, 0, x.n_rows(), x2);
	

	x.Destruct();
	la::TransposeInit(x3, &x); 

	population_second_stage_x_.Init(num_of_people_);
	start_col=0;
  for(index_t i=0; i<num_of_people_; i++) {
    population_second_stage_x_[i].Init(x.n_rows(), (index_t)info1.get(0,i));
	  population_second_stage_x_[i].CopyColumnFromMat(0, start_col, 
							(index_t)info1.get(0,i), x);
    start_col+=(index_t)info1.get(0,i);
  }

	/*cout<<"data file 2:";
	for(index_t i=0; i<num_of_people_; i++) {
		cout<<"population_second_stage_x["<<i<<"]"<<endl;
		for(index_t j=0; j<population_second_stage_x_[i].n_rows(); j++){
			for(index_t k=0; k<population_second_stage_x_[i].n_cols(); k++) {
				cout<<population_second_stage_x_[i].get(j,k)<<" ";
			}
			cout<<endl;
		}
		cout<<endl;		
	}
	*/

	//need to be fixed
	const char *data_file3=fx_param_str_req(module_, "data3");
  x.Destruct();
  data::Load(data_file3, &x);
  population_unknown_x_past_.Init(num_of_people_);
  start_col=0;
  for(index_t i=0; i<num_of_people_; i++) {
    population_unknown_x_past_[i].Init(x.n_rows(), (index_t)info1.get(0,i));
    population_unknown_x_past_[i].CopyColumnFromMat(0, start_col, 
        (index_t)info1.get(0,i), x);
    start_col+=(index_t)info1.get(0, i);
  }

	 
	const char *info_y_file=fx_param_str_req(module_, "info_y");
  Matrix info_y;
  data::Load(info_y_file, &info_y);
  population_first_stage_y_.Init(num_of_people_);
  for(index_t i=0; i<num_of_people_; i++) {
	  population_first_stage_y_[i]=(index_t)info_y.get(0,i);
  }

  const char *info_ind_unknown_x_file=fx_param_str_req(module_, "info_ind_unknown_x");
  Matrix info_ind_unknown_x;
  data::Load(info_ind_unknown_x_file, &info_ind_unknown_x);
  num_of_unknown_x_=info_ind_unknown_x.n_cols();
  population_ind_unknown_x_.Init(num_of_unknown_x_);
  for(index_t i=0; i<num_of_unknown_x_; i++) {
	  population_ind_unknown_x_[i]=info_ind_unknown_x.get(0,i);
  }

	ind_unknown_x->Copy(population_ind_unknown_x_);

	
	if (fx_param_exists(module_, "starting_points")) {
		const char *initial_parameter_file=fx_param_str_req(module_,"starting_points");
		Matrix mtx_initial_parameter;
		if(data::Load(initial_parameter_file, &mtx_initial_parameter)==SUCCESS_FAIL) {
			FATAL("File %s not found", initial_parameter_file);
		}
		else{

			if(mtx_initial_parameter.n_rows() != num_of_betas+2){
				NOTIFY("The number of starting points given is not same as the number of parameters");
				NOTIFY("Use default...");
				//initial_parameter->Destruct(); 
				initial_parameter->Init(num_of_betas+1);	//beta+p+q
				//initial_parameter->SetZero();
				//(*initial_parameter)[num_of_betas]=2;
				//(*initial_parameter)[num_of_betas+1]=2;
				initial_parameter->SetAll(1.5);

				
				
			}	//if
			else{
				mtx_initial_parameter.MakeColumnVector(0, initial_parameter);
				/*
				cout<<"Starting points are:   ";
				for(index_t i=0; i<initial_parameter->length(); i++){
					cout<<(*initial_parameter)[i]<<" ";
				}
				cout<<endl;
				*/
			}
		}	//else
	}	//if
	else {
		//NOTIFY("Starting points are not given. Use default...");
		//NOTIFY("Number of parameters is %d", num_of_betas+2);
		//cout<<"Number of parameters is "<<(num_of_betas+2)<<endl;
		//initial_parameter->Init(num_of_betas+2);
		
		//initial_parameter->SetAll(1.5);

		
		//initial_parameter->SetAll(0);
		
	  //cout<<"NUM_OF_BETAS="<<num_of_betas<<endl;
		//(*initial_parameter)[num_of_betas]=2;
		//(*initial_parameter)[num_of_betas+1]=2;
		


	}



	//const char *initial_parameter_file=fx_param_str(module_, "stating_points", "default_initial.csv");
	/*if(initial_parameter_file !=NULL) {
		Matrix mtx_initial_parameter;
    data::Load(initial_parameter_file, &mtx_initial_parameter);
		mtx_initial_parameter.MakeColumnVector(0, initial_parameter);
	}
	else {
		NOTIFY("Starting points are not given. Use default...");
		initial_parameter->Init(x.n_rows());
		initial_parameter->SetZero();
	}
	*/
	/*Matrix mtx_initial_parameter;
  data::Load(initial_parameter_file, &mtx_initial_parameter);
	if(mtx_initial_parameter.n_rows()==1 && mtx_initial_parameter.get(0,0)==0){
		NOTIFY("Starting points are not given. Use default...");
		initial_parameter->Init(x.n_rows());
		initial_parameter->SetZero();
	}
	else if(mtx_initial_parameter.n_rows() != x.n_rows()){
		FATAL("The number of starting points given is not same as the number of parameters");
	}
	else {
		mtx_initial_parameter.MakeColumnVector(0, initial_parameter);
	}
	*/


  double arg_initial_percent_sample=fx_param_double(module_, "initial_percent_sample", num_of_people_*0.05);

	//Initilize memeber variables
	num_of_selected_sample_=0;
	count_num_sampling_=0;
	initial_percent_sample_=arg_initial_percent_sample;
  *initial_percent_sample=initial_percent_sample_;

	shuffled_array_.Init(num_of_people_);

	//return num_of_people_;
	
	
  const char *opt_x_file=fx_param_str_req(module_, "opt_x");
  Matrix mtx_opt_x;
  data::Load(opt_x_file, &mtx_opt_x);

	Vector temp_opt_x;
	temp_opt_x.Init(mtx_opt_x.n_cols());

	for(index_t i=0; i<mtx_opt_x.n_cols(); i++){
		temp_opt_x[i]=mtx_opt_x.get(0,i);
	}
	  
	opt_x->Copy(temp_opt_x);
  //mtx_opt_x.MakeColumnVector(0, opt_x);
	//cout<<"n_rows="<<mtx_opt_x.n_rows()<<endl;
	//cout<<"n_cols="<<mtx_opt_x.n_cols()<<endl;

	cout<<"len="<<opt_x->length()<<endl;

	initial_parameter->Init(mtx_opt_x.n_rows()+2);
	initial_parameter->SetAll(1.5);

		

  
  

}



void DSampling::Shuffle() {
	//check - can be done in initilization
	int random =0;
	for(index_t i=0; i<num_of_people_; i++){
		shuffled_array_[i]=i;
	}	//i

	//http://www.cppreference.com/wiki/stl/algorithm/random_shuffle
//	std::random_shuffle(shuffled_array_.ptr(),
//	                  shuffled_array_.ptr()+num_of_people_);
		
   
	//Shuffle elements by randomly exchanging each with one other.
	for(index_t j=0; j<num_of_people_-1; j++){
		//random number for remaining position
		random = j+(rand() % (num_of_people_-j));	
		//shuffle
		swap( shuffled_array_[j], shuffled_array_[random] );
	}	//j

	/*
	cout<<"shuffled_array :";
	for(index_t i=0; i<num_of_people_; i++){
		cout<<shuffled_array_[i] <<" ";
	}
	cout<<endl;
	*/

	//num_of_people_=5000;
	//subset
	//num_of_people_=5000;


}

void DSampling::Shuffle2() {
	//check - can be done in initilization
	int random =0;
	for(index_t i=0; i<num_of_people_; i++){
		shuffled_array_[i]=i;
	}	//i

	//http://www.cppreference.com/wiki/stl/algorithm/random_shuffle
//	std::random_shuffle(shuffled_array_.ptr(),
//	                  shuffled_array_.ptr()+num_of_people_);
		
   
	//Shuffle elements by randomly exchanging each with one other.
	for(index_t j=0; j<num_of_people_-1; j++){
		//random number for remaining position
		random = j+(rand() % (num_of_people_-j));	
		//shuffle
		swap( shuffled_array_[j], shuffled_array_[random] );
	}	//j

	/*
	cout<<"shuffled_array :";
	for(index_t i=0; i<num_of_people_; i++){
		cout<<shuffled_array_[i] <<" ";
	}
	cout<<endl;
	*/

	//subset
	//change here for num_of_people
	//num_of_people_=5000;


}

/*void Sampling::ExpandSubset(double percent_added_sample, ArrayList<Matrix> *added_first_stage_x, 
											 ArrayList<Matrix> *added_second_stage_x, ArrayList<Matrix> *added_unknown_x_past, 
											 ArrayList<index_t> *added_first_stage_y, Vector *ind_unknown_x) {
												 */

void DSampling::ExpandSubset(double percent_added_sample, ArrayList<Matrix> *added_first_stage_x, 
											 ArrayList<Matrix> *added_second_stage_x, ArrayList<Matrix> *added_unknown_x_past, 
											 ArrayList<index_t> *added_first_stage_y) {
	
	int num_added_sample=0;
	
	//int old_num_selected_sample=num_of_selected_sample_;

	/*if(count_num_sampling_==0) {
		num_add_sample=math::RoundInt((num_of_people_)*(percent_added_sample));
		if(num_add_sample==0) {
			NOTIFY("number of sample to add is zero. start with Two samples.");
			num_add_sample=2;
		}

	}	else {
		num_add_sample=math::RoundInt((num_of_selected_sample_)*(percent_added_sample));
	}	//else
	*/
	if(num_of_selected_sample_==0) {
		NOTIFY("Initial sampling...");
		cout<<"Initial sampling..."<<endl;
		//cout<<"initial_percent_sample_="<<initial_percent_sample_<<endl;
		num_added_sample=math::RoundInt((num_of_people_)*(initial_percent_sample_)/100);
	} else {
		if(percent_added_sample==0 && (num_of_selected_sample_ < num_of_people_) ){
			NOTIFY("Keep the currect sample size");
			cout<<"Keep the current sample size? Add 100 samples"<<endl;
			num_added_sample=100;
		}
		else{
			num_added_sample=math::RoundInt((num_of_selected_sample_)*(percent_added_sample)/100);
			if(num_added_sample==0){
				NOTIFY("number of sample to add is zero. Add 100 samples.");
				cout<<"number of sample to add is zero. Add 100 samples."<<endl;
				num_added_sample=100;
			}

		}

	}
	

	//Copy to currect Subset
	//added_first_stage_x->Init();
	//added_second_stage_x->Init();
	//added_unknown_x_past->Init();
	//added_first_stage_y->Init();
	//ind_unknown_x->CopyValues(population_ind_unknown_x_);


	if(num_added_sample+num_of_selected_sample_ >= num_of_people_){
		for(index_t i=num_of_selected_sample_; i<num_of_people_; i++){
			added_first_stage_x->PushBackCopy(population_first_stage_x_[shuffled_array_[i]]);
			added_second_stage_x->PushBackCopy(population_second_stage_x_[shuffled_array_[i]]);
			added_unknown_x_past->PushBackCopy(population_unknown_x_past_[shuffled_array_[i]]);
			added_first_stage_y->PushBackCopy(population_first_stage_y_[shuffled_array_[i]]);
		}		//i	
		num_of_selected_sample_=num_of_people_;
		//NOTIFY("All data are used");
		
	} else {
		for(index_t i=num_of_selected_sample_; i<(num_of_selected_sample_+num_added_sample); i++){
			added_first_stage_x->PushBackCopy(population_first_stage_x_[shuffled_array_[i]]);
			added_second_stage_x->PushBackCopy(population_second_stage_x_[shuffled_array_[i]]);
			added_unknown_x_past->PushBackCopy(population_unknown_x_past_[shuffled_array_[i]]);
			added_first_stage_y->PushBackCopy(population_first_stage_y_[shuffled_array_[i]]);			
		}	//i
		num_of_selected_sample_+=num_added_sample;		
	}	//else
	
	//DEBUG_SAME_SIZE(sample_selector->size(), num_selected_sample);
  /*DEBUG_ASSERT_MSG(added_first_stage_x->size()==num_added_sample, 
      "Size of added first stage x is not same as number of added sample %i != %i",
      added_first_stage_x->size(), num_added_sample);
*/
	
	//cout<<"num_added_sample="<<num_added_sample<<endl;
	if(num_of_selected_sample_!=num_of_people_){
		count_num_sampling_+=1;
		NOTIFY("Extend subset (%d)", count_num_sampling_);
		cout<<"Extend subset "<<count_num_sampling_<<endl;
		NOTIFY("Add %d sample(s)", num_added_sample);
		cout<<"Add "<<num_added_sample<<" sample(s)"<<endl;
	}



		
			
	
}



/*double Sampling::CalculateSamplingError() {

}
*/

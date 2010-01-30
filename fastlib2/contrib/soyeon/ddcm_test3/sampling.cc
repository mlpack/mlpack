#include "sampling.h"
#include <stdlib.h>	//rand() srand()
#include <time.h>	//time()
#include <algorithm>	//swap
#include <iostream>
using namespace std;


void Sampling::Init(fx_module *module, int *num_of_people, 
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

			if(mtx_initial_parameter.n_rows() != num_of_betas+2){
				NOTIFY("The number of starting points given is not same as the number of parameters");
				NOTIFY("Use default...");
				//initial_parameter->Destruct(); 
				initial_parameter->Init(num_of_betas+2);	//beta+p+q
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
		NOTIFY("Starting points are not given. Use default...");
		//NOTIFY("Number of parameters is %d", num_of_betas+2);
		cout<<"Number of parameters is "<<(num_of_betas+2)<<endl;
		initial_parameter->Init(num_of_betas+2);
		
		//initial_parameter->SetAll(1.5);

		
		initial_parameter->SetAll(0);
		
	cout<<"NUM_OF_BETAS="<<num_of_betas<<endl;
		(*initial_parameter)[num_of_betas]=50;
		(*initial_parameter)[num_of_betas+1]=2;
		
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

		(*initial_parameter)[0]=1.3427;
		(*initial_parameter)[1]=1.63946;
		(*initial_parameter)[num_of_betas]=1.48997;
		(*initial_parameter)[num_of_betas+1]=0.968507;

		//new starting points
		(*initial_parameter)[0]=10;
		(*initial_parameter)[1]=5;
		(*initial_parameter)[num_of_betas]=9;
		(*initial_parameter)[num_of_betas+1]=3;

		(*initial_parameter)[0]=1.30307;
		(*initial_parameter)[1]=1.60892;
		(*initial_parameter)[num_of_betas]=8.12362;
		(*initial_parameter)[num_of_betas+1]=4.4049;

		(*initial_parameter)[0]=1.31451;
		(*initial_parameter)[1]=1.59058;
		(*initial_parameter)[num_of_betas]=7.61954;
		(*initial_parameter)[num_of_betas+1]=4.85318;

		//New starting points3
		(*initial_parameter)[0]=4;
		(*initial_parameter)[1]=2;
		(*initial_parameter)[num_of_betas]=1;
		(*initial_parameter)[num_of_betas+1]=5;

		(*initial_parameter)[0]=1.32692;
		(*initial_parameter)[1]=1.59923;
		(*initial_parameter)[num_of_betas]=4.06731;
		(*initial_parameter)[num_of_betas+1]=2.72315;

		(*initial_parameter)[0]=1.32494;
		(*initial_parameter)[1]=1.60465;
		(*initial_parameter)[num_of_betas]=3.76776;
		(*initial_parameter)[num_of_betas+1]=2.49188;

		(*initial_parameter)[0]=1.32688;
		(*initial_parameter)[1]=1.60729;
		(*initial_parameter)[num_of_betas]=3.45074;
		(*initial_parameter)[num_of_betas+1]=2.28265;

		(*initial_parameter)[0]=1.33014;
		(*initial_parameter)[1]=1.61029;
		(*initial_parameter)[num_of_betas]=3.09799;
		(*initial_parameter)[num_of_betas+1]=2.04948;

		(*initial_parameter)[0]=1.3292;
		(*initial_parameter)[1]=1.6166;
		(*initial_parameter)[num_of_betas]=2.72726;
		(*initial_parameter)[num_of_betas+1]=1.80404;


		(*initial_parameter)[0]=1.3351;
		(*initial_parameter)[1]=1.62064;
		(*initial_parameter)[num_of_betas]=2.35074;
		(*initial_parameter)[num_of_betas+1]=1.55331;

		(*initial_parameter)[0]=1.33445;
		(*initial_parameter)[1]=1.63047;
		(*initial_parameter)[num_of_betas]=1.94111;
		(*initial_parameter)[num_of_betas+1]=1.27843;
		
		//starting4-res4
		(*initial_parameter)[0]=-1;
		(*initial_parameter)[1]=0.8;
		(*initial_parameter)[num_of_betas]=0.2;
		(*initial_parameter)[num_of_betas+1]=2.1;

		(*initial_parameter)[0]=1.34142;
		(*initial_parameter)[1]=1.63586;
		(*initial_parameter)[num_of_betas]=1.62324;
		(*initial_parameter)[num_of_betas+1]=1.06105;

		(*initial_parameter)[0]=7;
		(*initial_parameter)[1]=1;
		(*initial_parameter)[num_of_betas]=2.5;
		(*initial_parameter)[num_of_betas+1]=0.3;

		(*initial_parameter)[0]=1.33617;
		(*initial_parameter)[1]=1.63394;
		(*initial_parameter)[num_of_betas]=1.78005;
		(*initial_parameter)[num_of_betas+1]=1.16897;

		//starting5-result5
		(*initial_parameter)[0]=2;
		(*initial_parameter)[1]=0.5;
		(*initial_parameter)[num_of_betas]=1;
		(*initial_parameter)[num_of_betas+1]=7;

		(*initial_parameter)[0]=1.32166;
		(*initial_parameter)[1]=1.57594;
		(*initial_parameter)[num_of_betas]=5.54512;
		(*initial_parameter)[num_of_betas+1]=4.22564;
		
		(*initial_parameter)[0]=1.31737;
		(*initial_parameter)[1]=1.59347;
		(*initial_parameter)[num_of_betas]=5.58899;
		(*initial_parameter)[num_of_betas+1]=3.72215;

		(*initial_parameter)[0]=1.31997;
		(*initial_parameter)[1]=1.59434;
		(*initial_parameter)[num_of_betas]=5.35786;
		(*initial_parameter)[num_of_betas+1]=3.53924;

		//sample=1000
    (*initial_parameter)[0]=6.16656;
		(*initial_parameter)[1]=19.1923;
		(*initial_parameter)[num_of_betas]=2.43075;
		(*initial_parameter)[num_of_betas+1]=3.30936;

		(*initial_parameter)[0]=7.85992;
		(*initial_parameter)[1]=24.3449;
		(*initial_parameter)[num_of_betas]=2.11071;
		(*initial_parameter)[num_of_betas+1]=2.81889;

		(*initial_parameter)[0]=8.85745;
		(*initial_parameter)[1]=27.3563;
		(*initial_parameter)[num_of_betas]=2.02756;
		(*initial_parameter)[num_of_betas+1]=2.5745;
		
    //sample=1000-2
    (*initial_parameter)[0]=8.64681;
		(*initial_parameter)[1]=27.5385;
		(*initial_parameter)[num_of_betas]=8.11649;
		(*initial_parameter)[num_of_betas+1]=7.56718;

		(*initial_parameter)[0]=13.1332;
		(*initial_parameter)[1]=40.8001;
		(*initial_parameter)[num_of_betas]=6.87032;
		(*initial_parameter)[num_of_betas+1]=6.4066;

		//simulation5_error
    (*initial_parameter)[0]=1.66669;
		(*initial_parameter)[1]=4.37995;
		(*initial_parameter)[num_of_betas]=2.88629;
		(*initial_parameter)[num_of_betas+1]=1.83085;

		//simulation5_error res3
		(*initial_parameter)[0]=1.17397;
		(*initial_parameter)[1]=3.42542;
		(*initial_parameter)[num_of_betas]=1.94735;
		(*initial_parameter)[num_of_betas+1]=2.702;

		//simulation5_error res4
		(*initial_parameter)[0]=1.43696;
		(*initial_parameter)[1]=3.97589;
		(*initial_parameter)[num_of_betas]=4.54396;
		(*initial_parameter)[num_of_betas+1]=3.64368;

		(*initial_parameter)[0]=1.4485;
		(*initial_parameter)[1]=3.92903;
		(*initial_parameter)[num_of_betas]=5.54422;
		(*initial_parameter)[num_of_betas+1]=4.3535;

		(*initial_parameter)[0]=1.43443;
		(*initial_parameter)[1]=3.91498;
		(*initial_parameter)[num_of_betas]=6.06644;
		(*initial_parameter)[num_of_betas+1]=4.76733;

		(*initial_parameter)[0]=1.80045;
		(*initial_parameter)[1]=6.16382;
		(*initial_parameter)[num_of_betas]=1.24462;
		(*initial_parameter)[num_of_betas+1]=0.779631;
		
		//simulation6-res2
		(*initial_parameter)[0]=2;
		(*initial_parameter)[1]=5;
		(*initial_parameter)[num_of_betas]=4;
		(*initial_parameter)[num_of_betas+1]=3;

		(*initial_parameter)[0]=1.4101;
		(*initial_parameter)[1]=5.65133;
		(*initial_parameter)[num_of_betas]=4.59492;
		(*initial_parameter)[num_of_betas+1]=4.58733;

		(*initial_parameter)[0]=1.39652;
		(*initial_parameter)[1]=5.53831;
		(*initial_parameter)[num_of_betas]=5.85271;
		(*initial_parameter)[num_of_betas+1]=5.8539;

		(*initial_parameter)[0]=1.718;
		(*initial_parameter)[1]=6.76955;
		(*initial_parameter)[num_of_betas]=6.0539;
		(*initial_parameter)[num_of_betas+1]=1.56463;

		(*initial_parameter)[0]=1.28105;
		(*initial_parameter)[1]=5.38665;
		(*initial_parameter)[num_of_betas]=4.50331;
		(*initial_parameter)[num_of_betas+1]=2.53439;

		(*initial_parameter)[0]=2;
		(*initial_parameter)[1]=5;
		(*initial_parameter)[num_of_betas]=4;
		(*initial_parameter)[num_of_betas+1]=3;

		(*initial_parameter)[0]=1.5982;
		(*initial_parameter)[1]=5.96623;
		(*initial_parameter)[num_of_betas]=5.17915;
		(*initial_parameter)[num_of_betas+1]=1.22117;

		
		(*initial_parameter)[0]=1.43698;
		(*initial_parameter)[1]=6.09037;
		(*initial_parameter)[num_of_betas]=2.75128;
		(*initial_parameter)[num_of_betas+1]=0.644982;

		(*initial_parameter)[0]=2;
		(*initial_parameter)[1]=5;
		(*initial_parameter)[num_of_betas]=4;
		(*initial_parameter)[num_of_betas+1]=3;

		(*initial_parameter)[0]=-1.40239;
		(*initial_parameter)[1]=2.56635;
		(*initial_parameter)[num_of_betas]=4.28058;
		(*initial_parameter)[num_of_betas+1]=1.55379;

		(*initial_parameter)[0]=-1.39473;
		(*initial_parameter)[1]=2.5818;
		(*initial_parameter)[num_of_betas]=3.55608;
		(*initial_parameter)[num_of_betas+1]=1.32611;

		(*initial_parameter)[0]=1.52416;
		(*initial_parameter)[1]=-2.09689;
		(*initial_parameter)[num_of_betas]=4.62156;
		(*initial_parameter)[num_of_betas+1]=1.7312;

		(*initial_parameter)[0]=1;
		(*initial_parameter)[1]=1;
		(*initial_parameter)[num_of_betas]=1;
		(*initial_parameter)[num_of_betas+1]=1;

		(*initial_parameter)[0]=4.21104;
		(*initial_parameter)[1]=-3.08401;
		(*initial_parameter)[num_of_betas]=4.09364;
		(*initial_parameter)[num_of_betas+1]=0.670095;

		(*initial_parameter)[0]=1;
		(*initial_parameter)[1]=1;
		(*initial_parameter)[num_of_betas]=3;
		(*initial_parameter)[num_of_betas+1]=2;

		(*initial_parameter)[0]=3.1008;
		(*initial_parameter)[1]=-2.32536;
		(*initial_parameter)[num_of_betas]=6.50685;
		(*initial_parameter)[num_of_betas+1]=0.444984;

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

void Sampling::Init2(fx_module *module, int *num_of_people, 
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
				initial_parameter->Init(num_of_betas+2);	//beta+p+q
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
		NOTIFY("Starting points are not given. Use default...");
		//NOTIFY("Number of parameters is %d", num_of_betas+2);
		cout<<"Number of parameters is "<<(num_of_betas+2)<<endl;
		initial_parameter->Init(num_of_betas+2);
		
		//initial_parameter->SetAll(1.5);

		
		initial_parameter->SetAll(0);
		
	cout<<"NUM_OF_BETAS="<<num_of_betas<<endl;
		(*initial_parameter)[num_of_betas]=10;
		(*initial_parameter)[num_of_betas+1]=10;
		
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

		(*initial_parameter)[0]=1.3427;
		(*initial_parameter)[1]=1.63946;
		(*initial_parameter)[num_of_betas]=1.48997;
		(*initial_parameter)[num_of_betas+1]=0.968507;

		//new starting points
		(*initial_parameter)[0]=10;
		(*initial_parameter)[1]=5;
		(*initial_parameter)[num_of_betas]=9;
		(*initial_parameter)[num_of_betas+1]=3;

		(*initial_parameter)[0]=1.30307;
		(*initial_parameter)[1]=1.60892;
		(*initial_parameter)[num_of_betas]=8.12362;
		(*initial_parameter)[num_of_betas+1]=4.4049;

		(*initial_parameter)[0]=1.31451;
		(*initial_parameter)[1]=1.59058;
		(*initial_parameter)[num_of_betas]=7.61954;
		(*initial_parameter)[num_of_betas+1]=4.85318;

		//New starting points3
		(*initial_parameter)[0]=4;
		(*initial_parameter)[1]=2;
		(*initial_parameter)[num_of_betas]=1;
		(*initial_parameter)[num_of_betas+1]=5;

		(*initial_parameter)[0]=1.32692;
		(*initial_parameter)[1]=1.59923;
		(*initial_parameter)[num_of_betas]=4.06731;
		(*initial_parameter)[num_of_betas+1]=2.72315;

		(*initial_parameter)[0]=1.32494;
		(*initial_parameter)[1]=1.60465;
		(*initial_parameter)[num_of_betas]=3.76776;
		(*initial_parameter)[num_of_betas+1]=2.49188;

		(*initial_parameter)[0]=1.32688;
		(*initial_parameter)[1]=1.60729;
		(*initial_parameter)[num_of_betas]=3.45074;
		(*initial_parameter)[num_of_betas+1]=2.28265;

		(*initial_parameter)[0]=1.33014;
		(*initial_parameter)[1]=1.61029;
		(*initial_parameter)[num_of_betas]=3.09799;
		(*initial_parameter)[num_of_betas+1]=2.04948;

		(*initial_parameter)[0]=1.3292;
		(*initial_parameter)[1]=1.6166;
		(*initial_parameter)[num_of_betas]=2.72726;
		(*initial_parameter)[num_of_betas+1]=1.80404;


		(*initial_parameter)[0]=1.3351;
		(*initial_parameter)[1]=1.62064;
		(*initial_parameter)[num_of_betas]=2.35074;
		(*initial_parameter)[num_of_betas+1]=1.55331;

		(*initial_parameter)[0]=1.33445;
		(*initial_parameter)[1]=1.63047;
		(*initial_parameter)[num_of_betas]=1.94111;
		(*initial_parameter)[num_of_betas+1]=1.27843;
		
		//starting4-res4
		(*initial_parameter)[0]=-1;
		(*initial_parameter)[1]=0.8;
		(*initial_parameter)[num_of_betas]=0.2;
		(*initial_parameter)[num_of_betas+1]=2.1;

		(*initial_parameter)[0]=1.34142;
		(*initial_parameter)[1]=1.63586;
		(*initial_parameter)[num_of_betas]=1.62324;
		(*initial_parameter)[num_of_betas+1]=1.06105;

		(*initial_parameter)[0]=7;
		(*initial_parameter)[1]=1;
		(*initial_parameter)[num_of_betas]=2.5;
		(*initial_parameter)[num_of_betas+1]=0.3;

		(*initial_parameter)[0]=1.33617;
		(*initial_parameter)[1]=1.63394;
		(*initial_parameter)[num_of_betas]=1.78005;
		(*initial_parameter)[num_of_betas+1]=1.16897;

		//starting5-result5
		(*initial_parameter)[0]=2;
		(*initial_parameter)[1]=0.5;
		(*initial_parameter)[num_of_betas]=1;
		(*initial_parameter)[num_of_betas+1]=7;

		(*initial_parameter)[0]=1.32166;
		(*initial_parameter)[1]=1.57594;
		(*initial_parameter)[num_of_betas]=5.54512;
		(*initial_parameter)[num_of_betas+1]=4.22564;
		
		(*initial_parameter)[0]=1.31737;
		(*initial_parameter)[1]=1.59347;
		(*initial_parameter)[num_of_betas]=5.58899;
		(*initial_parameter)[num_of_betas+1]=3.72215;

		(*initial_parameter)[0]=1.31997;
		(*initial_parameter)[1]=1.59434;
		(*initial_parameter)[num_of_betas]=5.35786;
		(*initial_parameter)[num_of_betas+1]=3.53924;

		//sample=1000
    (*initial_parameter)[0]=6.16656;
		(*initial_parameter)[1]=19.1923;
		(*initial_parameter)[num_of_betas]=2.43075;
		(*initial_parameter)[num_of_betas+1]=3.30936;

		(*initial_parameter)[0]=7.85992;
		(*initial_parameter)[1]=24.3449;
		(*initial_parameter)[num_of_betas]=2.11071;
		(*initial_parameter)[num_of_betas+1]=2.81889;

		(*initial_parameter)[0]=8.85745;
		(*initial_parameter)[1]=27.3563;
		(*initial_parameter)[num_of_betas]=2.02756;
		(*initial_parameter)[num_of_betas+1]=2.5745;
		
    //sample=1000-2
    (*initial_parameter)[0]=8.64681;
		(*initial_parameter)[1]=27.5385;
		(*initial_parameter)[num_of_betas]=8.11649;
		(*initial_parameter)[num_of_betas+1]=7.56718;

		(*initial_parameter)[0]=13.1332;
		(*initial_parameter)[1]=40.8001;
		(*initial_parameter)[num_of_betas]=6.87032;
		(*initial_parameter)[num_of_betas+1]=6.4066;

		//simulation5_error
    (*initial_parameter)[0]=1.66669;
		(*initial_parameter)[1]=4.37995;
		(*initial_parameter)[num_of_betas]=2.88629;
		(*initial_parameter)[num_of_betas+1]=1.83085;

		//simulation5_error res3
		(*initial_parameter)[0]=1.17397;
		(*initial_parameter)[1]=3.42542;
		(*initial_parameter)[num_of_betas]=1.94735;
		(*initial_parameter)[num_of_betas+1]=2.702;

		//simulation5_error res4
		(*initial_parameter)[0]=1.43696;
		(*initial_parameter)[1]=3.97589;
		(*initial_parameter)[num_of_betas]=4.54396;
		(*initial_parameter)[num_of_betas+1]=3.64368;

		(*initial_parameter)[0]=1.4485;
		(*initial_parameter)[1]=3.92903;
		(*initial_parameter)[num_of_betas]=5.54422;
		(*initial_parameter)[num_of_betas+1]=4.3535;

		(*initial_parameter)[0]=1.43443;
		(*initial_parameter)[1]=3.91498;
		(*initial_parameter)[num_of_betas]=6.06644;
		(*initial_parameter)[num_of_betas+1]=4.76733;

		(*initial_parameter)[0]=1.80045;
		(*initial_parameter)[1]=6.16382;
		(*initial_parameter)[num_of_betas]=1.24462;
		(*initial_parameter)[num_of_betas+1]=0.779631;
		
		//simulation6-res2
		(*initial_parameter)[0]=2;
		(*initial_parameter)[1]=5;
		(*initial_parameter)[num_of_betas]=4;
		(*initial_parameter)[num_of_betas+1]=3;

		(*initial_parameter)[0]=1.4101;
		(*initial_parameter)[1]=5.65133;
		(*initial_parameter)[num_of_betas]=4.59492;
		(*initial_parameter)[num_of_betas+1]=4.58733;

		(*initial_parameter)[0]=1.39652;
		(*initial_parameter)[1]=5.53831;
		(*initial_parameter)[num_of_betas]=5.85271;
		(*initial_parameter)[num_of_betas+1]=5.8539;

		(*initial_parameter)[0]=1.718;
		(*initial_parameter)[1]=6.76955;
		(*initial_parameter)[num_of_betas]=6.0539;
		(*initial_parameter)[num_of_betas+1]=1.56463;

		(*initial_parameter)[0]=1.28105;
		(*initial_parameter)[1]=5.38665;
		(*initial_parameter)[num_of_betas]=4.50331;
		(*initial_parameter)[num_of_betas+1]=2.53439;

		(*initial_parameter)[0]=2;
		(*initial_parameter)[1]=5;
		(*initial_parameter)[num_of_betas]=4;
		(*initial_parameter)[num_of_betas+1]=3;

		(*initial_parameter)[0]=1.5982;
		(*initial_parameter)[1]=5.96623;
		(*initial_parameter)[num_of_betas]=5.17915;
		(*initial_parameter)[num_of_betas+1]=1.22117;

		
		(*initial_parameter)[0]=1.43698;
		(*initial_parameter)[1]=6.09037;
		(*initial_parameter)[num_of_betas]=2.75128;
		(*initial_parameter)[num_of_betas+1]=0.644982;

		(*initial_parameter)[0]=2;
		(*initial_parameter)[1]=5;
		(*initial_parameter)[num_of_betas]=4;
		(*initial_parameter)[num_of_betas+1]=3;

		(*initial_parameter)[0]=-1.40239;
		(*initial_parameter)[1]=2.56635;
		(*initial_parameter)[num_of_betas]=4.28058;
		(*initial_parameter)[num_of_betas+1]=1.55379;

		(*initial_parameter)[0]=-1.39473;
		(*initial_parameter)[1]=2.5818;
		(*initial_parameter)[num_of_betas]=3.55608;
		(*initial_parameter)[num_of_betas+1]=1.32611;

		(*initial_parameter)[0]=1.52416;
		(*initial_parameter)[1]=-2.09689;
		(*initial_parameter)[num_of_betas]=4.62156;
		(*initial_parameter)[num_of_betas+1]=1.7312;

		(*initial_parameter)[0]=1;
		(*initial_parameter)[1]=1;
		(*initial_parameter)[num_of_betas]=1;
		(*initial_parameter)[num_of_betas+1]=1;

		(*initial_parameter)[0]=4.21104;
		(*initial_parameter)[1]=-3.08401;
		(*initial_parameter)[num_of_betas]=4.09364;
		(*initial_parameter)[num_of_betas+1]=0.670095;

		(*initial_parameter)[0]=1;
		(*initial_parameter)[1]=1;
		(*initial_parameter)[num_of_betas]=3;
		(*initial_parameter)[num_of_betas+1]=2;

		(*initial_parameter)[0]=3.1008;
		(*initial_parameter)[1]=-2.32536;
		(*initial_parameter)[num_of_betas]=6.50685;
		(*initial_parameter)[num_of_betas+1]=0.444984;

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

void Sampling::Init3(fx_module *module, int *num_of_people, 
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
				initial_parameter->Init(num_of_betas+2);	//beta+p+q
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



void Sampling::Shuffle() {
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

	
	cout<<"shuffled_array :";
	for(index_t i=0; i<num_of_people_; i++){
		cout<<shuffled_array_[i] <<" ";
	}
	cout<<endl;
	
	//num_of_people_=5000;
	//subset
	//num_of_people_=5000;


}

void Sampling::Shuffle2() {
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

	
	cout<<"shuffled_array :";
	for(index_t i=0; i<num_of_people_; i++){
		cout<<shuffled_array_[i] <<" ";
	}
	cout<<endl;
	
	//subset
//	num_of_people_=5000;


}

/*void Sampling::ExpandSubset(double percent_added_sample, ArrayList<Matrix> *added_first_stage_x, 
											 ArrayList<Matrix> *added_second_stage_x, ArrayList<Matrix> *added_unknown_x_past, 
											 ArrayList<index_t> *added_first_stage_y, Vector *ind_unknown_x) {
												 */

void Sampling::ExpandSubset(double percent_added_sample, ArrayList<Matrix> *added_first_stage_x, 
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

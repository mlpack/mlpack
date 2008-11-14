#include "sampling.h"
#include <stdlib.h>	//rand() srand()
#include <time.h>	//time()
#include <algorithm>	//swap
#include <iostream>
using namespace std;



void Sampling::Init(fx_module *module) {
	module_=module;
	const char *data_file1=fx_param_str_req(module_, "data1");
	const char *info_file1=fx_param_str_req(module_, "info1");
	Matrix x;
	data::Load(data_file1, &x);
	//num_of_betas_=x.n_rows();
	
  Matrix info1;
  data::Load(info_file1, &info1);
	num_of_people_=info1.n_cols();
  population_first_stage_x_.Init(num_of_people_);
	index_t start_col=0;
  for(index_t i=0; i<num_of_people_; i++) {
    population_first_stage_x_[i].Init(x.n_rows(), (index_t)info1.get(0, i));
    population_first_stage_x_[i].CopyColumnFromMat(0, start_col, 
										(index_t)info1.get(0,i), x);
    start_col+=(index_t)info1.get(0, i);
  }
	
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

  double initial_percent_sample=fx_param_double(module_, "initial_percent_sample", 5);

	//Initilize memeber variables
	num_of_selected_sample_=0;
	count_num_sampling_=0;
	initial_percent_sample_=initial_percent_sample;

	shuffled_array_.Init(num_of_people_);

}



void Sampling::Shuffle_() {
	//check - can be done in initilization
	int random =0;
	

	//Vector shuffled_array_;
	//shuffled_array_.Init(num_of_people_);

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

}



void Sampling::ExpandSubset(double percent_added_sample, ArrayList<Matrix> *added_first_stage_x, 
											 ArrayList<Matrix> *added_second_stage_x, ArrayList<Matrix> *added_unknown_x_past, 
											 ArrayList<index_t> *added_first_stage_y, Vector *ind_unknown_x) {
		
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
		num_added_sample=math::RoundInt((num_of_people_)*(percent_added_sample)/100);
	} else {
		num_added_sample=math::RoundInt((num_of_selected_sample_)*(percent_added_sample)/100);
	}

	if(num_added_sample==0) {
			NOTIFY("number of sample to add is zero. start with Two samples.");
			num_added_sample=2;
	}	//if



	//Copy to currect Subset
	added_first_stage_x->Init();
	added_second_stage_x->Init();
	added_unknown_x_past->Init();
	added_first_stage_y->Init();
	ind_unknown_x->Copy(population_ind_unknown_x_);


	if(num_added_sample+num_of_selected_sample_ >= num_of_people_){
		for(index_t i=num_of_selected_sample_; i<num_of_people_; i++){
			added_first_stage_x->PushBackCopy(population_first_stage_x_[shuffled_array_[i]]);
			added_second_stage_x->PushBackCopy(population_second_stage_x_[shuffled_array_[i]]);
			added_unknown_x_past->PushBackCopy(population_unknown_x_past_[shuffled_array_[i]]);
			added_first_stage_y->PushBackCopy(population_first_stage_y_[shuffled_array_[i]]);
		}		//i	
		num_of_selected_sample_=num_of_people_;
		NOTIFY("All data are used");
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
  DEBUG_ASSERT_MSG(added_first_stage_x->size()==num_added_sample, 
      "Size of added first stage x is not same as number of added sample %i != %i",
      added_first_stage_x->size(), num_added_sample);

	count_num_sampling_+=1;
	NOTIFY("This is %d th sampling", count_num_sampling_);
	cout<<"num_added_sample="<<num_added_sample<<endl;
	
}


/*double Sampling::CalculateSamplingError() {

}
*/


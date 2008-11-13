#include "sampling.h"
#include <stdlib.h>	//rand() srand()
#include <time.h>	//time()
#include <algorithm>	//swap



Sampling::Init(fx_module *module) {
	num_slected_sample_=0;
	count_num_sampling_=0;

}



Sampling::Shuffle() {
	//check - can be done in initilization
	num_people_=first_stage_x_.size();
	int random =0;
	int temp=0;

	//Vector shuffled_array_;
	//shuffled_array_.Init(num_people_);

	for(index_t i=0; i<num_people_; i++){
		shuffled_array_[i]=i;
	}	//i

	//http://www.cppreference.com/wiki/stl/algorithm/random_shuffle
//	std::random_shuffle(shuffled_array_.ptr(),
//	                  shuffled_array_.ptr()+num_of_people_);
		
   
	//Shuffle elements by randomly exchanging each with one other.
	for(index_t j=0; j<num_people_-1; j++){
		//random number for remaining position
		random = j+(rand() % (num_people_-j));	
		//shuffle
		swap( shuffled_array_[j], shuffled_array_[random] );
	}	//j

}



Sampling::InitialSampling(ArrayList<Matrix> *initial_first_stage_x, ArrayList<Matrix> *initial_second_stage_x, 
													ArrayList<Matrix> *initial_unknown_x_past, ArrayList<index_t> *initial_first_stage_y, 
													ArrayList<index_t> *ind_unknown_x) {
	//Vector sample_selector;
	//sample_selector.Init(num_initial_sampling_);
	for(index_t i=0; i<num_initial_sampling_; i++){
			//check
			//sample_selector_[i]=shuffled_array[i];
			//check-what's the difference?
			sample_selector_.PushBackCopy(shuffled_array[i]);
			
	}	//i
	num_slected_sample_=num_initial_sampling_;
	//DEBUG_SAME_SIZE(sample_selector->size(), num_selected_sample);
  DEBUG_ASSERT_MSG(sample_sepector_.size()==num_selected_sample, 
      "Size of sample selector is not same as number of selected sample %i != %i",
      sample_sepector_.size(), num_selected_sample);

	//Copy to currect Subset
	ArrayList<Matrix> initial_first_stage_x_temp;
	first_stage_x_temp.Init(num_selected_sample_);

	ArrayList<Matrix> initial_second_stage_x_temp;
	second_stage_x_temp.Init(num_selected_sample_);

	ArrayList<Matrix> initial_unknown_x_past_temp;
	unknown_x_past_temp.Init(num_selected_sample_);

	ArrayList<index_t> initial_first_stage_y_temp;
	first_stage_y_temp.Init(num_selected_sample_);

	//ArrayList<index_t> initial_second_stage_y_temp;
	//second_stage_Y_temp.Init(num_selected_sample);

	//ArrayList<index_t> ind_unknown_x_temp;
	//ind_unknown_x_temp.Init(num_of_unknown_x_);

	ind_unknown_x->Copy(population_ind_unknown_x_);

	for(index_t i=0; i<num_selected_sample; i++){
		initial_first_stage_x_temp[i].Alias(population_first_stage_x_(sample_selector[i]);
		initial_second_stage_x_temp[i].Alias(population_second_stage_x_(sample_selector[i]);
		initial_unknown_x_past_temp[i].Alias(population_unknown_x_past_(sample_selector[i]);
		initial_first_stage_y_temp[i].Alias(population_first_stage_y_(sample_selector[i]);
	}

	initial_first_stage_x->Copy(initial_first_stage_x_temp);
	initial_second_stage_x->Copy(initial_second_stage_x_temp);
	initial_unknown_x_past->Copy(initial_unknown_x_past_temp);
	initial_first_stage_y->Copy(initial_first_stage_y_temp);

	count_num_sampling_+=1;
	
	}		//i


}


Sampling::ExpandSubset(double percent_added_sample, ArrayList<Matrix> *added_first_stage_x, 
											 ArrayList<Matrix> *added_second_stage_x, ArrayList<Matrix> *added_unknown_x_past, 
											 ArrayList<index_t> *added_first_stage_y, Vector *ind_unknown_x) {
		
	int num_add_sample=0;
	//int old_num_selected_sample=num_selected_sample_;

	if(count_num_sampling_==0) {
		num_add_sample=math::RoundInt((num_people_)*(percent_added_sample));
	}	else {
		num_add_sample=math::RoundInt((num_selected_sample_)*(percent_added_sample));
	}	//else

	//Copy to currect Subset
	added_first_stage_x.Init();
	added_second_stage_x.Init();
	added_unknown_x_past.Init();
	added_first_stage_y.Init();
	ind_unknown_x->Copy(population_ind_unknown_x_);


	if(num_add_sample+num_selected_sample_ >= num_people_){
		for(index_t i=num_selected_sample_; i<num_people_; i++){
			added_first_stage_x.PushBackCopy(population_first_stage_x_(sample_selector[i]);
			added_second_stage_x.PushBackCopy(population_second_stage_x_(sample_selector[i]);
			added_unknown_x_past.PushBackCopy(population_unknown_x_past_(sample_selector[i]);
			added_first_stage_y.PushBackCopy(population_first_stage_y_(sample_selector[i]);
		}		//i	
		num_selected_sample_=num_people_;
	} else {
		for(index_t i=num_selected_sample_; i<(num_selected_sample_+num_add_sample); i++){
			added_first_stage_x.PushBackCopy(population_first_stage_x_(sample_selector[i]);
			added_second_stage_x.PushBackCopy(population_second_stage_x_(sample_selector[i]);
			added_unknown_x_past.PushBackCopy(population_unknown_x_past_(sample_selector[i]);
			added_first_stage_y.PushBackCopy(population_first_stage_y_(sample_selector[i]);			
		}	//i
		num_selected_sample_+=num_add_sample;		
	}	//else
	
	//DEBUG_SAME_SIZE(sample_selector->size(), num_selected_sample);
  DEBUG_ASSERT_MSG(added_first_stage_x->size()==num_added_sample, 
      "Size of added first stage x is not same as number of added sample %i != %i",
      added_first_stage_x->size(), num_added_sample);

	count_num_sampling_+=1;
	
}


/*Sampling::CalculateSamplingError(double *sigma_n){

}
*/


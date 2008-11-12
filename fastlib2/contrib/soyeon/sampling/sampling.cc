#include "sampling.h"
#include <stdlib.h>	//rand() srand()
#include <time.h>	//time()
#include <algorithm>	//swap



Sampling::Init(fx_module *module) {
	//ind_initial_sampling_=0;

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


	//Shuffle elements by randomly exchanging each with one other.
	for(index_t j=0; j<num_people_-1; j++){
		//random number for remaining position
		random = j+(rand() % (num_people_-j));	
		//shuffle
		swap( shuffled_array_[j], shuffled_array_[random] );
	}	//j

}



Sampling::InitialSampling(ArrayList<index_t> *sample_selector, double *num_slected_sample) {
	for(index_t i=0; i<num_initial_sampling; i++){
			//check
			//sample_selector_[i]=shuffled_array[i];
			//check-what's the difference?
			sample_selector->PushBackCopy(shuffled_array[i]);
			*num_slected_sample=num_initial_sampling;
		}	//i

}



Sampling::ExpandSubset(double percent_added_sample, ArrayList<index_t> *sample_selector, double *num_selected_sample) {
	
	int num_add_sample=0;
	num_add_sample=math::RoundInt((num_selected_sample_)*(percent_added_sample));
	if(num_add_sample+num_selected_sample_ >= num_people_){
		for(index_t i=num_selected_sample_; i<num_people_; i++){
			sample_selector->PushbackCopy(shuffled_array[i]);
			//sample_selector.size()==num_seleted_sample_==num_people_
			*num_selected_sample=num_people_;
		}	//i
	} else {
		for(index_t i=num_selected_sample_; i<(num_selected_sample_+num_add_sample); i++){
			sample_selector->PushBackCopy(shuffled_array[i]);
			//sample_selector.size()==num_seleted_sample_
			*num_selected_sample+=num_add_sample;
		}	//i
	}	//else
	
}



Sampling::ReturnSubsetData(double num_selected_sample, ArrayList<index_t> &sample_selector, 
													 ArrayList<Matrix> *added_first_stage_x, ArrayList<Matrix> *added_second_stage_x, 
													 ArrayList<Matrix> *added_unknown_x_past, ArrayList<index_t> *added_first_stage_y, 
													 ArrayList<index_t> *added_second_stage_y, ArrayList<index_t> *ind_unknown_x) {

	//assume input files are saved as desired form
	int num_selected_people;
	num_selected_people=sample_selector.size();

	ArrayList<Matrix> added_first_stage_x_temp;
	first_stage_x_temp.Init(num_selected_people);

	ArrayList<Matrix> added_second_stage_x_temp;
	second_stage_x_temp.Init(num_selected_people);

	ArrayList<Matrix> added_unknown_x_past_temp;
	unknown_x_past_temp.Init(num_selected_people);

	ArrayList<index_t> added_first_stage_y_temp;
	first_stage_y_temp.Init(num_selected_people);

	ArrayList<index_t> added_second_stage_y_temp;
	second_stage_Y_temp.Init(num_selected_people);

	ArrayList<index_t> ind_unknown_x_temp;
	ind_unknown_x_temp.Init(num_of_unknown_x_);
	
	//check - change to Vector and Alias and copy one time
	for(index_t i=0; i<num_of_unknown_x_; i++){
		ind_unknown_x_temp[i]=population_ind_unknown_x_[i];
	}



	for(index_t i=0; i<num_selected_people; i++){
		added_first_stage_x_temp[i].Alias(population_first_stage_x_(sample_selector[i]);


	
	}		//i


	MakeColumnVector
}




Sampling::CalculateSamplingError(double *sigma_n){

}


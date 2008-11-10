#include "sampling.h"
#include <stdlib.h>	//rand() srand()
#include <time.h>	//time()
#include <algorithm>	//swap



Sampling::Init(fx_module *module) {
	ind_initial_sampling_=0;

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

Sampling::ExpandSubset(double percent_added_sample, ArrayList<index_t> *sample_selector_) {
	if(ind_initial_sampling_==0) {
		//copy num_initial_sampling 
		for(index_t i=0; i<num_initial_sampling; i++){
			//check
			//sample_selector_[i]=shuffled_array[i];
			//check-what's the difference?
			sample_selector_.PushBackCopy(shuffled_array[i]);
		}	//i
		ind_initial_sampling_=1;
		num_slected_sample_=num_initial_sampling;
	}	else {
			int num_add_sample=0;
			num_add_sample=math::RoundInt((num_selected_sample_)*(percent_added_sample));
			if(num_add_sample+num_selected_sample_ >= num_people_){
				for(index_t i=num_selected_sample_; i<num_people_; i++){
					sample_selector_.PushbackCopy(shuffled_array[i]);
					num_selected_sample_=num_people_;
				}	//i
			} else {
				for(index_t i=num_selected_sample_; i<(num_selected_sample_+num_add_sample); i++){
					sample_selector_.PushBackCopy(shuffled_array[i]);
					num_selected_sample_+=num_add_sample;
				}	//i
			}	//else
	}	//else
	
}

//Next-->function for picking the data from selected people 
Sampling::ReturnSubsetData(ArrayList<Matrix> Input_file1, ArrayList<Matrix> Input_file2, 
													 ArrayList<Matrix> Input_file3) {

	MakeColumnVector
}



Sampling::CalculateSamplingError(double *sigma_n){

}

void CalculateSamplingError(ml_data *dt,double *sigma_n)
{
   int i,count=0;
   double correctionfactor;
   double temp,sum=0.0,sum_sq=0.0;
   
   //////////////////////////////////////////
   // References...
   int np=dt->npersons;		
   int *person_selector=dt->person_selector;
   double *person_SP=dt->person_SP;
   double *person_SP_new=dt->person_SP_new;
   //////////////////////////////////////////

   for(i=0;i<np;i++)
   {
      if((person_selector)&&(person_selector[i]==0))
	 continue;
      count++;
      temp=log(person_SP[i]/person_SP_new[i]);
      sum+=temp;
      sum_sq+=(temp*temp);
   }
   correctionfactor=sqrt(((double)(np-count))/(np-1));
   sum_sq-=((sum*sum)/count);
   sum_sq/=count;
   sum_sq/=(count-1);
   
   (*sigma_n)=correctionfactor*sqrt(sum_sq);
}
		






}




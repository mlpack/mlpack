#include "fastlib/fastlib.h"


class Sampling {
	public:
	 void Init(fx_module *module);
	 void InitialSampling(ArrayList<index_t> *sample_selector, double *num_selected_sample)
	 void ExpandSubset(double percent_added_sample, ArrayList<index_t> *sample_selector, double *num_selected_sample);
	 void ReturnSubsetData(ArrayList<index_t> &sample_selector, 
													 ArrayList<Matrix> *first_stage_x, ArrayList<Matrix> *second_stage_x,
													 ArrayList<Matrix> *unknown_x_past, ArrayList<index_t> *first_stage_y,
													 ArrayList<index_t> *second_stage_y, ArrayList<index_t> *ind_unknown_x);

	private:
		
		ArrayList<Matrix> population_first_stage_x_;
		ArrayList<Matrix> population_second_stage_x_;
		ArrayList<Matrix> population_unknown_x_past_;
		ArrayList<index_t> population_first_stage_y_;
		//ArrayList<index_t> population_second_stage_y_;
		Vector population_ind_unknown_x_;

		//ArrayList<Matrix> sample_first_stage_x_;
		//ArrayList<Matrix> sample_second_stage_x_;
		//ArrayList<Matrix> sample_unknown_x_past_;
		//ArrayList<index_t> sample_first_stage_y_;
		


		int num_of_unknown_x_;



		int num_people_;		//whole population
		ArrayList<index_t> shuffled_array_; //length=num_people_
		//ArrayList<index_t> sample_selector_;

		//1 if initial_sampling is done, then need to expand_subset
		//0 if initial_sampling is not done yet
		//int ind_initial_sampling_;	
		int initial_percent_sample_; //one of the argument
		int num_slected_sample_;
		int count_num_sampling_;
		


		//ArrayList<index_t> sample_selector_; //
		
		void Shuffle();
		void CalculateSamplingError(double *sigma_n);


};







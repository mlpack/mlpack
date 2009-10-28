#include "fastlib/fastlib.h"

//class SamplingTest;
class Sampling {
	//friend class SamplingTest;
	public:
	 
	 void Init(fx_module *module, int *num_of_people, 
											Vector *ind_unknown_x, 
											double *initial_percent_sample,
											Vector *initial_parameter);
	 /*void ExpandSubset(double percent_added_sample, ArrayList<Matrix> *added_first_stage_x, 
										 ArrayList<Matrix> *added_second_stage_x, ArrayList<Matrix> *added_unknown_x_past, 
										 ArrayList<index_t> *added_first_stage_y, Vector *ind_unknown_x);
	 */
	 //model with intercept
	 void Init2(fx_module *module, int *num_of_people, 
											Vector *ind_unknown_x, 
											double *initial_percent_sample,
											Vector *initial_parameter);
	 void Shuffle();
	 void Shuffle2();
	 void ExpandSubset(double percent_added_sample, ArrayList<Matrix> *added_first_stage_x, 
										 ArrayList<Matrix> *added_second_stage_x, ArrayList<Matrix> *added_unknown_x_past, 
										 ArrayList<index_t> *added_first_stage_y);
	 //void ExpandSubset2(double percent_added_sample);
		//double CalculateSamplingError();

	private:

		fx_module *module_;		
		//Objective objective;
		ArrayList<Matrix> population_first_stage_x_;
		ArrayList<Matrix> population_second_stage_x_;
		ArrayList<Matrix> population_unknown_x_past_;
		ArrayList<index_t> population_first_stage_y_;
		//ArrayList<index_t> population_second_stage_y_;
		Vector population_ind_unknown_x_;

		int num_of_unknown_x_;
		int num_of_people_;		//whole population
		ArrayList<index_t> shuffled_array_; //length=num_of_people_
		//ArrayList<index_t> sample_selector_;		
		double initial_percent_sample_; //one of the argument
		int num_of_selected_sample_;
		int count_num_sampling_;

		
		


};











#include "fastlib/fastlib.h"


class Objective {
 public:
  void Init(fx_module *module);
  void ComputeObjective(Matrix &x, double *value);
 
 private:
  ArrayList<Matrix> first_stage_x_; 
	ArrayList<Matrix> second_stage_x_; 
	//first_stage_x_.size()==second_stage_x_.size()
	//ArrayList<Matrix> second_stage_x_tilde_; //known attributes
	//ArrayList<Matrix> second_stage_x_bar_; //unknown attributes
	
	//Information for exponential smoothing
	ArrayList<Matrix> unk_x_past_;
	//x_bar with max_number of alternatives for all people
	//nrow=num_unk_x_, ncol=max_number_alternatives
	Matrix max_size_x_bar; 
	
	
  
  ArrayList<index_t>  first_stage_y_;
	// If the value is -1 then it corresponds
  // to all zeros in y
  // If it is greter than zero then it corresponds to 
  // the non zero element index
	ArrayList<index_t> second_stage_y_;
	//1 then it corresponds to all one in y^post
	
	//it corresponds to the unknown attributes index in unk_x_past file
	//(eg. 7th attribute(price) is unknown 
	ArrayList<index_t> ind_unk_x_;


	ArrayList<index_t> exp_betas_times_x1_;
	ArrayList<index_t> exp_betas_times_x2_;
	//ArrayList<index_t> exp_betas_times_x2_tilde_;
	ArrayList<index_t> postponed_probability_;
	//ArrayList<index_t> conditional_postponed_probability_;

	//max num of choices among all
	//unk_x_past_.size==max_number_alternatives_
	int max_number_alternatives_;
	//number of positive elements in ind_unk_x
	int num_unk_x_;

	
	double denumerator_beta_function_;
	int num_of_t_beta_fn_;
	double t_weight_;
	int num_of_alphas_;
	double alpha_wieght_;

  double ComputeTerm1_(Vector &betas);
  double ComputeTerm2_(Vector &betas, double p, double q);
  double ComputeTerm3_();
	void ComputePostponedProbability_(Vector &betas, double p, double q);
	void ComputeExpBetasTimesX1_(Vector &betas);
	//void ComputeExpBetasTimesX2_tilde_(Vector &betas);
	void ComputeDeumeratorBetaFunction_(double p, doulbe q);
	//void ComputeMaxSizeXBar_();

	
	


	
};


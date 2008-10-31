#include "fastlib/fastlib.h"


class Objective {
 public:
  void Init(fx_module *module);
  void ComputeObjective(Matrix &x, double *value);
 
 private:
  ArrayList<Matrix> first_stage_x_; 
	ArrayList<Matrix> second_stage_x_; 
	//first_stage_x_.size()==second_stage_x_.size()
	
  
  ArrayList<index_t>  first_stage_y_;
	// If the value is -1 then it corresponds
  // to all zeros in y
  // If it is greter than zero then it corresponds to 
  // the non zero element index
	ArrayList<index_t> second_stage_y_;
	//1 then it corresponds to all one in y^post

	ArrayList<index_t> exp_betas_times_x1_;
	ArrayList<index_t> postponed_probability_;

	int num_of_alphas_;

  double ComputeTerm1_(Vector &betas);
  double ComputeTerm2_(Vector &betas, double p, double q);
  double ComputeTerm3_();
	void ComputePostponedProbability_(Vector &betas, double p, double q);
	void ComputeExpBetasTimesX1_(Vector &betas);

	ArrayList<index_t> exp_betas_times_x1_;
	ArrayList<index_t> postponed_probability_;
	
};


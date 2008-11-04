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

///////////////////////////////////////////////////////

	//add new things from here for objective2 (Compute gradient)
	ArrayList<Matrix> first_stage_dot_logit_;
	ArrayList<Matrix> first_stage_ddot_logit_;

	ArrayList<Matrix> second_stage_dot_logit_;
	//ArrayList<Matrix> second_stage_ddot_logit_;

	//ArrayList<index_t> derivative_beta_conditional_postponed_prob_;
	//ArrayList<index_t> conditional_postponed_prob_;

	ArrayList<index_t> sum_first_derivative_conditional_postpond_prob_;
	ArrayList<index_t> sum_second_derivative_conditional_postpond_prob_;


	//ArrayList<index_t> SumSecondDerivativeConditionalPostpondProb_

	//need exp_betas_times_x1 and exp_betas_times_x2
	void ComputeDotLogit_(Vector &betas);

	//need DotLogit
	void ComputeDDotLogit_( );
	//need DotLogit
	Vector ComputeDerivativeBetaTerm1_();
	Vector ComputeDerivativeBetaTerm2_();
	Vector ComputeDerivativeBetaTerm3_();

	Matrix ComputeSecondDerivativeBetaTerm1_();
	Matrix ComputeSecondDerivativeBetaTerm2_();
	Matrix ComputeSecondDerivativeBetaTerm3_();
	//void ComputeDerivativeBetaConditionalPostponedProb_(Vector &betas);
	
	//need first_stage_dot_logit_
	void ComputeSumDerivativeConditionalPostpondProb_(Vector &betas);

	double ComputeDerivativePTerm1_();
	double ComputeDerivativePTerm2_();
	double ComputeDerivativePTerm3_();

	double ComputeSecondDerivativePTerm1_();
	double ComputeSecondDerivativePTerm2_();
	double ComputeSecondDerivativePTerm3_();

	double ComputeDerivativeQTerm1_();
	double ComputeDerivativeQTerm2_();
	double ComputeDerivativeQTerm3_();

	double ComputeSecondDerivativeQTerm1_();
	double ComputeSecondDerivativeQTerm2_();
	double ComputeSecondDerivativeQTerm3_();


	void ComputeSumDerivativeBetaFunction_(Vector &betas, double p, double q);

	ArrayList<index_t> sum_first_derivative_p_beta_fn_;
	ArrayList<index_t> sum_second_derivative_p_beta_fn;
	ArrayList<index_t> sum_first_derivative_q_beta_fn_;
	ArrayList<index_t> sum_second_derivative_q_beta_fn_;
	ArrayList<index_t> sum_second_derivative_p_q_beta_fn_;
	ArrayList<index_t> sum_second_derivative_conditionl_postponed_p_;
	ArrayList<index_t> sum_second_derivative_conditionl_postponed_q_;
	

	Vector ComputeSecondDerivativePBetaTerm1_();
	Vector ComputeSecondDerivativePBetaTerm2_();
	Vector ComputeSecondDerivativePBetaTerm3_();

	Vector ComputeSecondDerivativeQBetaTerm1_();
	Vector ComputeSecondDerivativeQBetaTerm2_();
	Vector ComputeSecondDerivativeQBetaTerm3_();

	double ComputeSecondDerivativePQTerm1_();
	double ComputeSecondDerivativePQTerm2_();
	double ComputeSecondDerivativePQTerm3_();











	




	
};





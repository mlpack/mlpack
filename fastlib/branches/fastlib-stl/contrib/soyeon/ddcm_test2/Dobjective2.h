#include "fastlib/fastlib.h"

//class Sampling;
class DObjective {
//friend class Sampling;
 public:
	//Vector current_parameter;
  /*void Init(ArrayList<Matrix> &added_first_stage_x,
						ArrayList<Matrix> &added_second_stage_x, 
						ArrayList<Matrix> &added_unknown_x_past, 
						ArrayList<index_t> &added_first_stage_y,
						Vector &ind_unknown_x);
  */
	void Init2(Vector &ind_unknown_x, int count_init2);
	void Init3(int sample_size,
						ArrayList<Matrix> &added_first_stage_x,
						ArrayList<Matrix> &added_second_stage_x, 
						ArrayList<Matrix> &added_unknown_x_past, 
						ArrayList<index_t> &added_first_stage_y);
						
						

	//void Destruct();
  void ComputeObjective(double current_sample,
												Vector &current_parameter, 
												double *objective);
	void ComputeGradient(double current_sample,
											 Vector &current_parameter, 
										   Vector *gradient);
	void ComputeHessian(double current_sample,
											Vector &current_parameter, 
										  Matrix *hessian);
	void ComputeChoiceProbability(Vector &current_parameter, 
																				 Vector *choice_probability);
	void CheckGradient(double current_sample,
										 Vector &current_parameter, 
										 Vector *approx_gradient);
	void CheckHessian(double current_sample, 
									  Vector &current_parameter, 
										Matrix *approx_hessian);

	void CheckHessian2(double current_sample, 
									  Vector &current_parameter, 
										Matrix *approx_hessian);

	void CheckHessian3(double current_sample, 
									  Vector &current_parameter, 
										Matrix *approx_hessian);
	void ComputePredictionError(double current_sample, 
									  Vector &current_parameter,
										ArrayList<index_t> &true_decision,
										double *postponed_prediction_error,
										double *choice_prediction_error);

 
 private:
  fx_module *module_;

	ArrayList<Matrix> first_stage_x_; 
	ArrayList<Matrix> second_stage_x_; 
	//first_stage_x_.size()==second_stage_x_.size()
	//ArrayList<Matrix> second_stage_x_tilde_; //known attributes
	//ArrayList<Matrix> second_stage_x_bar_; //unknown attributes
	
	//Information for exponential smoothing
	ArrayList<Matrix> unknown_x_past_;
	//x_bar with max_number of alternatives for all people
	//nrow=num_unk_x_, ncol=max_number_alternatives
	
  
  ArrayList<index_t>  first_stage_y_;
	// If the value is -1 then it corresponds
  // to all zeros in y
  // If it is greter than zero then it corresponds to 
  // the non zero element index
	//ArrayList<index_t> second_stage_y_;
	//1 then it corresponds to all one in y^post
	
	//it corresponds to the unknown attributes index in unk_x_past file
	//(eg. 7th attribute(price) is unknown 
	Vector ind_unknown_x_;


	ArrayList<double> exp_betas_times_x1_;
	ArrayList<double> exp_betas_times_x2_;
	
  ArrayList<Matrix> first_derivative_second_stage_x_;
	ArrayList<Matrix> Second_derivative_second_stage_x_;

	//ArrayList<index_t> exp_betas_times_x2_tilde_;
	ArrayList<double> postponed_probability_;
	//ArrayList<index_t> conditional_postponed_probability_;

	//max num of choices among all
	//unk_x_past_.size==max_number_alternatives_
	//int max_number_alternatives_;
	//number of positive elements in ind_unk_x
	//int num_unknown_x_;
  index_t num_of_betas_;	
	
	/*
	double denumerator_beta_function_;
	index_t num_of_t_beta_fn_;
  double t_weight_;
	index_t num_of_alphas_;
	double alpha_weight_;
  */

  double ComputeTerm1_(Vector &betas);
  double ComputeTerm2_();
  double ComputeTerm3_();
	//void ComputePostponedProbability_(Vector &betas, double p, double q);
	void ComputePostponedProbability_(Vector &betas, double alpha);
	
	void ComputeExpBetasTimesX1_(Vector &betas);
	////void ComputeExpBetasTimesX2_tilde_(Vector &betas);
	//void ComputeDeumeratorBetaFunction_(double p, double q);
	////void ComputeMaxSizeXBar_();


///////////////////////////////////////////////////////
////////calculate gradient
/////////////////////////////////////////////////////////////


	//add new things from here for objective2 (Compute gradient)
	ArrayList<Vector> first_stage_dot_logit_;
	ArrayList<Matrix> first_stage_ddot_logit_;

	ArrayList<Vector> second_stage_dot_logit_;
	ArrayList<Matrix> second_stage_ddot_logit_;

	//ArrayList<index_t> derivative_beta_conditional_postponed_prob_;
	//ArrayList<index_t> conditional_postponed_prob_;

	//ArrayList<Vector> sum_first_derivative_conditional_postpond_prob_;
	//ArrayList<Matrix> sum_second_derivative_conditional_postpond_prob_;
  ArrayList<Vector> first_derivative_postpond_prob_beta_;
  ArrayList<Matrix> second_derivative_postpond_prob_beta_;
  
  ArrayList<double> first_derivative_postpond_prob_alpha_;
	ArrayList<double> second_derivative_postpond_prob_alpha_;

	ArrayList<Vector> derivative_postpond_prob_alpha_beta_;

	//ArrayList<index_t> SumSecondDerivativeConditionalPostpondProb_

	//need exp_betas_times_x1 and exp_betas_times_x2
	void ComputeDotLogit_(Vector &betas);

	//need DotLogit
	void ComputeDDotLogit_();
	//need DotLogit
	void ComputeDerivativeBetaTerm1_(Vector *beta_term1);
	void ComputeDerivativeBetaTerm2_(Vector *beta_term2);
	void ComputeDerivativeBetaTerm3_(Vector *beta_term3);

	double ComputeDerivativeAlphaTerm1_();
	double ComputeDerivativeAlphaTerm2_();
	double ComputeDerivativeAlphaTerm3_();
	
	void ComputeDerivativePostpondProb_(Vector &betas, double alpha);
	//void ComputeDerivativePostpondProb_alpha_(Vector &betas, double alpha);
	
	//void ComputeDerivativeBetaConditionalPostponedProb_(Vector &betas);
	
	//need first_stage_dot_logit_
	
	/*
	void ComputeSumDerivativeConditionalPostpondProb_(Vector &betas, double p, double q);

	double ComputeDerivativePTerm1_();
	double ComputeDerivativePTerm2_();
	double ComputeDerivativePTerm3_();

	double ComputeDerivativeQTerm1_();
	double ComputeDerivativeQTerm2_();
	double ComputeDerivativeQTerm3_();

	void ComputeSumDerivativeBetaFunction_(Vector &betas, double p, double q);

	ArrayList<double> sum_first_derivative_p_beta_fn_;
	ArrayList<double> sum_second_derivative_p_beta_fn_;
	ArrayList<double> sum_first_derivative_q_beta_fn_;
	ArrayList<double> sum_second_derivative_q_beta_fn_;
	ArrayList<double> sum_second_derivative_p_q_beta_fn_;
	ArrayList<Vector> sum_second_derivative_conditionl_postponed_p_;
	ArrayList<Vector> sum_second_derivative_conditionl_postponed_q_;

	*/


	//Hessian
	void ComputeSecondDerivativeBetaTerm1_(Matrix *second_beta_term1);
	void ComputeSecondDerivativeBetaTerm2_(Matrix *second_beta_term2);
	void ComputeSecondDerivativeBetaTerm3_(Matrix *second_beta_term3);

	double ComputeSecondDerivativeAlphaTerm1_();
	double ComputeSecondDerivativeAlphaTerm2_();
	double ComputeSecondDerivativeAlphaTerm3_();


	void ComputeSecondDerivativeAlphaBetaTerm1_(Vector *alpha_beta_term1);
	void ComputeSecondDerivativeAlphaBetaTerm2_(Vector *alpha_beta_term2);
	void ComputeSecondDerivativeAlphaBetaTerm3_(Vector *alpha_beta_term3);

	/*
	double ComputeSecondDerivativePTerm1_();
	double ComputeSecondDerivativePTerm2_();
	double ComputeSecondDerivativePTerm3_();

	double ComputeSecondDerivativeQTerm1_();
	double ComputeSecondDerivativeQTerm2_();
	double ComputeSecondDerivativeQTerm3_();

	void ComputeSecondDerivativePBetaTerm1_(Vector *p_beta_term1);
	void ComputeSecondDerivativePBetaTerm2_(Vector *p_beta_term2);
	void ComputeSecondDerivativePBetaTerm3_(Vector *p_beta_term3);

	void ComputeSecondDerivativeQBetaTerm1_(Vector *q_beta_term1);
	void ComputeSecondDerivativeQBetaTerm2_(Vector *q_beta_term2);
	void ComputeSecondDerivativeQBetaTerm3_(Vector *q_beta_term3);

	double ComputeSecondDerivativePQTerm1_();
	double ComputeSecondDerivativePQTerm2_();
	double ComputeSecondDerivativePQTerm3_();

	*/



	
};








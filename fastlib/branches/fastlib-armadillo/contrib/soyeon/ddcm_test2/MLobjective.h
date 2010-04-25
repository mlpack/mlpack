#include "fastlib/fastlib.h"

//class Sampling;
class MLObjective {
//friend class Sampling;
 public:
	
	void Init2(int count_init2);
	void Init3(int sample_size,
						ArrayList<Matrix> &added_first_stage_x,
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
	 
  ArrayList<index_t>  first_stage_y_;
	// If the value is -1 then it corresponds
  // to all zeros in y
  // If it is greter than zero then it corresponds to 
  // the non zero element index
	//ArrayList<index_t> second_stage_y_;
	//1 then it corresponds to all one in y^post
	
	//it corresponds to the unknown attributes index in unk_x_past file
	//(eg. 7th attribute(price) is unknown 
	

	ArrayList<double> exp_betas_times_x1_;
		

	//ArrayList<index_t> exp_betas_times_x2_tilde_;
	
	//max num of choices among all
	//unk_x_past_.size==max_number_alternatives_
	//int max_number_alternatives_;
	//number of positive elements in ind_unk_x
	//int num_unknown_x_;
  index_t num_of_betas_;	
	

  double ComputeTerm1_(Vector &betas);
  
	void ComputeExpBetasTimesX1_(Vector &betas);
	//void ComputeExpBetasTimesX2_tilde_(Vector &betas);
	


///////////////////////////////////////////////////////
////////calculate gradient
/////////////////////////////////////////////////////////////


	//add new things from here for objective2 (Compute gradient)
	ArrayList<Vector> first_stage_dot_logit_;
	ArrayList<Matrix> first_stage_ddot_logit_;

	
	//need exp_betas_times_x1 and exp_betas_times_x2
	void ComputeDotLogit_(Vector &betas);

	//need DotLogit
	void ComputeDDotLogit_();
	//need DotLogit
	void ComputeDerivativeBetaTerm1_(Vector *beta_term1);
	

	
	//void ComputeDerivativeBetaConditionalPostponedProb_(Vector &betas);
	
	//need first_stage_dot_logit_
	///void ComputeSumDerivativeConditionalPostpondProb_(Vector &betas, double p, double q);

	
	
	
	//Hessian
	void ComputeSecondDerivativeBetaTerm1_(Matrix *second_beta_term1);
	
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








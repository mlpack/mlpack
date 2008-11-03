#include "objective.h"
#include <cmath>
    
Objective::Init(fx_module *module) {

}

Objective::ComputeObjective(Matrix &x, double *objective) {
  *objective = ComputeTerm1_() + ComputeTerm2_() + ComputeTerm3_();
}

double Objective::ComputeTerm1_(Vector &betas) {
  double term1=0;
  for(index_t n=0; n<first_stage_x_.size(); n++) {
    if (first_stage_y_[n]<0) { 
			//first_stage_y_[n]=-1 if all==zero, j_i is n chose j_i
      continue;
    } else {
      Vector temp;
      first_stage_x_[n].MakeColumnVector(first_stage_y_[n], &temp);
			term1+=la::Dot(betas, temp) - log(exp_betas_times_x1_[n]);
    }
  }
  return term1;
}

double Objective::ComputeTerm2_(Vector &betas, double p, double q) {
  double term2=0;
  for(index_t n=0; n<first_stage_x_.size(); n++) {
    if (first_stage_y_[n]<0) {
      continue;
    } else {
      DEBUG_ASSERT(1-postponed_probability_[n]);
      term2+=log(1-postponed_probability_[n]);
    }
  }
  return term2;
}

double Objecitve::ComputeTerm3_() {
  double term3=0;
  for(index_t n=0; n<first_stage_x_.size(); n++) {
    if (second_stage_y_[n]<0) {
      continue;
    } else {
      DEBUG_ASSERT(postponed_probability_[n]>0);
      term3+=log(postponed_probability_[n]);
    }
  }
  return term3;
}

//Compute x^2_{ni}(alpha), beta'x^2_{ni}(alpha), and postponedprob.
void Objective::ComputePostponedProbability_(Vector &betas, 
                                             double p, 
                                             double q) {
  postponed_probability_.SetZero(); 

	double alpha_temp=0;
	double beta_function_temp=0;
	double numerator=0;
	//need to specify
	num_of_alphas_=10;
	alpha_weight_=1/num_of_alphas;

	exp_betas_times_x2_.SetZero();
  
	for(index_t n=0; n<first_stage_x_.size; n++){
		for(index_t l=0; l<num_of_alphas; ;++){
			alpha_temp=(l+1)*(alpha_weight_);
		
			beta_function_temp=pow(alpha_temp, p-1)*pow((1-alpha_temp), q-1)/denumerator_beta_function_;

		
			//Calculate x^2_{ni}(alpha_l)
			for(index_t i=0; i<first_stage_x_[n].n_cols(); i++){
				int count=0;
				for(index_t j=ind_unk_x_[0]; j<ind_unk_x_[ind_unk_x_.size()]; j++){
					count+=1;
					exponential_temp=alpha_temp*first_stage_x_[n].get(i, j)
													+(alpha_temp)*(1-alpha_temp)*unk_x_past[i].get(count-1,1)
													+(alpha_temp)*pow((1-alpha_temp),2)*unk_x_past[i].get(count-1,2);
					second_stage_x_[n].set(j, i, exponential_temp);
				}	//j
			}	//i

			for(index_t i=0; i<second_stage_x_[n].n_cols(); i++) {
				exp_betas_times_x2_[n]+=exp(la::Dot(betas.size(), betas.ptr(),
											 second_stage_x_[n].GetColumnPtr(i) ));
			}
			//conditional_postponed_probability_[n]
			postponed_probability_[n]+=( (exp_betas_times_x2_[n]/(exp_betas_times_x1_[n]
																  + exp_betas_times_x2_[n]) )
																*beta_function_temp );			
		}	//alpha
		postponed_probability_[n]*=alpha_wieght_;	
	}	//n

}

void Objective::ComputeExpBetasTimesX1_(Vector &betas) {
  exp_betas_times_x1_.SetZero();
  //double sum=0;
	for(index_t n=0; n<first_stage_x_.size(); n++){
		for(index_t j=0; j<first_stage_x_[n].n_cols(); j++) {
			exp_betas_times_x1_[n]+=exp(la::Dot(betas.length(), 
															beta.ptr(), 
															first_stage_y_[n].GetColumnPtr(j)));
		}
  }
}


void Objective::ComputeDeumeratorBetaFunction_(double p, doulbe q) {
	denumerator_beta_function_=0;
	//Need to choose number of t points to approximate integral
	num_of_t_beta_fn_=10;
	double t_weight_=1/(num_of_t_beta_fn);
	double t_temp;
	for(index t tnum=0; tnum<num_of_t_beta_fn_; tnum++){
		t_temp=(tnum+1)*(t_weight_);

		//double pow( double base, double exp );
		denumerator_beta_function_+=pow(temp, p-1)*pow((1-t_temp), q-1);
	}
	denumerator*=(t_weight_);
}

//////////////////////////////////////////////////////////
//add new things from here for objective2 (Compute gradient) 
//Compute dot_logit
void Objective::ComputeDotLogit_(Vector &betas) {
	for(index_t n=0; n<first_stage_x_.size(); n++){
		for(index_t i=0; i<first_stage_x_.n_cols(); i++){
			first_stage_dot_logit_[n].set(i, 1, exp(la::Dot(betas.length(), beta.ptr(),
																				 (first_stage_x_[n].GetColumnPtr(i))/
																				 exp_betas_times_x1_[n]));
		}	//i
		/*for(index_t j=0; j<second_stage_x_.n_cols(); j++){
			second_stage_dot_logit_[n].set(j, 1, exp(la::Dot(betas.length(), beta.ptr(),
																				 (second_stage_x_[n].GetColumnPtr(i))/
																				 exp_betas_times_x2_[n]));
																				 
		}	//j
		*/
	}	//n
}


void Objective::ComputeDDotLogit_() {
	first_stage_ddot_logit_.SetZero();
	second_stage_ddot_logit_.SetZero();

	for(index_t n=0; n<first_stage_x_.size(); n++){
		for(index_t i=0; i<first_stage_x_.n_cols(); i++){
			first_stage_ddot_logit_[n].set(i, i, first_stage_dot_logit_[n].get(i,1));
		}	//i
		/*for(index_t j=0; i<second_stage_x_.n_cols(); j++){
			second_stage_ddot_logit_[n].set(j, j, second_stage_dot_logit_[n].get(j,1));
		}	//j*/

	}	//n

}


Vector Objective::ComputeDerivativeBetaTerm1_() {
	Vector derivative_beta_term1;
	derivative_beta_term1.Init(betas.length());
	derivative_beta_term1.SetZero();

  for(index_t n=0; n<first_stage_x_.size(); n++) {
    if (first_stage_y_[n]<0) { 
			//first_stage_y_[n]=-1 if all==zero, j_i is n chose j_i
      continue;
    } else {
      Vector temp;
			temp.Init(betas.length());
			la::MulOverwrite(first_stage_x_[n], first_stage_dot_logit[n]), &temp);
			la::SubOverwrite(first_stage_x_[n].GetColumnPtr(first_stage_y_[n]), &temp);
			//check
			la::Addto(derivative_beta_term1, &temp);
																							
		}
  }
  return derivative_beta_term1;
}


/*void Objective::ComputeDerivativeBetaConditionalPostponedProb_(Vector &betas){
	derivative_beta_conditional_postponed_prob_.SetZero();
	conditional_postponed_prob_.SetZero();

	//Vector temp;
	//temp.Init(betas.lentgh());
	//temp.SetZero();

	for(index_t n=0; n<first_stage_x_[n].size; n++){
		
		conditional_postponed_prob_[n]=exp_betas_times_x2_[n]/(exp_betas_times_x1_[n]+exp_betas_times_x2_[n]);
		DEBUG_ASSERT(conditional_postponed_prob_[n]>0);
		DEBUG_ASSERT(conditional_postponed_prob_[n]<1);
		la::MulOverwrite(first_stage_x_[n], first_stage_dot_logit[n]), &derivative_beta_conditional_postponed_prob_[n]);
		//check
		la::MulExpert(1, second_stage_x_[n], second_stage_dot_logit[n], -1, &derivative_beta_conditional_postponed_prob_[n]);
		la::scale( conditional_postponed_prob_*(1-conditional_postponed_prob_), &derivative_beta_conditional_postponed_prob_[n]);
		//derivative_beta_conditional_postponed_prob_[n]=&temp;
	}
}
*/

void Objective::ComputeSumDerivativeConditionalPostpondProb_(Vector &betas){
	
	Vector temp;
	temp.Init(betas.length());	//dotX1*dotLogit1
	//SumSecondDerivativeConditionalPostpondProb_.SetZero();

	double alpha_temp=0;
	double beta_function_temp=0;
	double numerator=0;
	//need to specify
	num_of_alphas_=10;
	alpha_weight_=1/num_of_alphas;

	exp_betas_times_x2_.SetZero();
	second_stage_dot_logit_.SetZero();
	double conditional_postponed_prob=0;
	Vector first_derivative_conditional_postpond_prob;
	first_derivative_conditional_postpond_prob.Init(betas.length());
	Vector temp2;	//dotX2*dotLogit2
	temp2.Init(betas.length());




	for(index_t n=0; n<first_stage_x_.size(); n++){
		la::MulOverwrite(first_stage_x_[n], first_stage_dot_logit_[n], &temp);
		sum_first_derivative_conditional_postpond_prob_[n].SetZero();

	 for(index_t l=0; l<num_of_alphas; ;++){
			alpha_temp=(l+1)*(alpha_weight_);
		
			beta_function_temp=pow(alpha_temp, p-1)*pow((1-alpha_temp), q-1)/denumerator_beta_function_;

		
			//Calculate x^2_{ni}(alpha_l)
			for(index_t i=0; i<first_stage_x_[n].n_cols(); i++){
				int count=0;
				for(index_t j=ind_unk_x_[0]; j<ind_unk_x_[ind_unk_x_.size()]; j++){
					count+=1;
					exponential_temp=alpha_temp*first_stage_x_[n].get(i, j)
													+(alpha_temp)*(1-alpha_temp)*unk_x_past[i].get(count-1,1)
													+(alpha_temp)*pow((1-alpha_temp),2)*unk_x_past[i].get(count-1,2);
					second_stage_x_[n].set(j, i, exponential_temp);
				}	//j
			}	//i

			for(index_t i=0; i<second_stage_x_[n].n_cols(); i++) {
				exp_betas_times_x2_[n]+=exp(la::Dot(betas.size(), betas.ptr(),
											 second_stage_x_[n].GetColumnPtr(i) ));
				//Calculate second_stage_dot_logit_
				second_stage_dot_logit_[n].set( i, 1, exp(la::Dot(betas.length(), beta.ptr(),
																				 (second_stage_x_[n].GetColumnPtr(i))/
																				 exp_betas_times_x2_[n]) );
				second_stage_ddot_logit_[n].set(i, i, first_stage_dot_logit_[n].get(i,1));
			}	//i
			conditional_postponed_prob=exp_betas_times_x2_[n]/(exp_betas_times_x1_[n]+exp_betas_times_x2_[n]);
			la::MulOverwrite(second_stage_x_[n], second_stage_dot_logit_[n], &temp2);
			la::SubOverwrite(temp2, temp1, &first_derivative_conditional_postpond_prob);

			//Calculate SecondDerivativePostponedProb.
			Matrix first_term_temp;
			first_term_temp.Init(betas.length(), betas.length());
			la::MulTransBOverwrite(first_derivative_conditional_postpond_prob, first_derivative_conditional_postpond_prob, 
														 &first_term_temp);
			la::Scale( (1-2*conditional_postponed_prob)*(conditional_postponed_prob)*(1-conditional_postponed_prob),
								&first_term_temp);

			//check
			Matrix temp3; //dotLogit*dotLogit'
			temp3.Init(second_stage_x_[n].n_cols(), second_stage_x_[n].n_cols());
			la::MulTransBOverwrite(second_stage_dot_logit_[n], second_stage_dot_logit_[n], &temp3);
			
			Matrix temp4; //ddotLogit-dotLogit*dotLogit'
			temp3.Init(second_stage_x_[n].n_cols(), second_stage_x_[n].n_cols());
			la::SubOverwrite(temp3, second_stage_ddot_logit_[n], &temp4);


			//ddotLogit-dotLogit*dotLogit'





			
			la::Scale( (conditional_postponed_prob)*(1-conditional_postponed_prob), &first_derivative_conditional_postpond_prob);

			
			//Scale with beta_function
					la::Scale( beta_function_temp, &first_derivative_conditional_postpond_prob );

			//Check
			la::Addto(sum_first_derivative_conditional_postpond_prob_[n], &first_derivative_conditional_postpond_prob);
					
		}	//alpha
		la::Scale(alpha_wieght_, &sum_first_derivative_conditional_postpond_prob_[n]);	
	}	//n

}



Vector Objective::ComputeDerivativeBetaTerm2_() {

	derivative_beta_term2.Init(betas.length());
	derivative_beta_term2.SetZero();
	Vector temp;
	temp.Init(betas.length());

	for(index_t n=0; n<first_stage_x_.size(); n++){
		if (first_stage_y_[n]<0) {
      continue;
    } else {
			//check
			temp=SumFirstDerivativeConditionalPostpondProb_[n]/(1-postponed_probability_[n]);
			//check
			la::Addto(derivative_beta_term2, &temp);

		}	//if-else

	}	//n
	return derivative_beta_term2;

}

Matrix Objective::ComputeSecondDerivativeBetaTerm1_() {
	//check
	Matrix second_derivative_beta_term1;
	second_derivative_beta_term1.Init(betas.length(), betas.length());
	derivative_beta_term1.SetAll(0.0);

  for(index_t n=0; n<first_stage_x_.size(); n++) {
    if (first_stage_y_[n]<0) { 
			//first_stage_y_[n]=-1 if all==zero, j_i is n chose j_i
      continue;
    } else {

			//check from here
      Vector temp1;
			temp1.Init(betas.length());
			la::MulOverwrite(first_stage_x_[n], first_stage_dot_logit[n]), &temp1);

			Matrix temp2
			temp2.Init(betas.length(), betas.length());
			la::MulTransBOverwrite(temp1, temp1, &temp2);

			Matrix temp3;
			temp3.Init(betas.length(), first_stage_x_[n].n_cols());
			la::MulOverwrite(first_stage_x_[n], first_stage_ddot_logit_[n], &temp3);

			Matrix temp4;
			temp3.Init(betas.length(), betas.length());
			la::MulTransBOverwrite(temp3, first_stage_x_[n], &temp4);
			//check
			la::SubFrom(temp4, &temp2);
			la::Addto(second_derivative_beta_term1, &temp2);
																							
		}
  }
  return second_derivative_beta_term1;
}
}



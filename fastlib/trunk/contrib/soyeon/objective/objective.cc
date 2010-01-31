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
				second_stage_x_[n]+=exp(la::Dot(betas.size(), betas.ptr(),
											 second_stage_x_[n].GetColumnPtr(i) ));
			}
			//conditional_postponed_probability_[n]
			postponed_probability_[n]+=( (second_stage_x_[n]/(exp_betas_times_x1_[n]
																	+ second_stage_x_[n]))
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


void Objective::ComputeDeumeratorBetaFunction_(double p, doulbe q){
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
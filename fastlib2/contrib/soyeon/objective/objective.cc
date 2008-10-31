#include "objective.h"

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
  for(index_t n=0; n<first_stage_x_.size(); n++) {
    for(index_t l=0; l<num_of_alphas_; l++) {
      double numerator=0;
      for(index_t i=0; i<second_stage_x_[n].n_cols(); i++) {
        nominator+=exp(la::Dot(betas.size(), betas.ptr(),
															 second_stage_y_[n].GetColumnPtr(i) ));
      }
      postponed_probability[n]+=numerator/(exp_betas_times_x1_+ numerator);
    }
    posponed_probability_*=alpha_wieght_;
  }
}

void ComputeExpBetasTimesX1_(Vector &betas) {
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

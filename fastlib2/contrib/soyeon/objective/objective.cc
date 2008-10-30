#include "objective.h"

Objective::Init(fx_module *module) {

}

Objective::ComputeObjective(Matrix &x, double *value) {
  *value = ComputeTerm1_() + ComputeTerm2_() + ComputeTerm3_();
}

double Objective::ComputeTerm1_(Vector &betas) {
  double term1=0;
  for(index_t n=0; n<first_stage_x_.size(); n++) {
    if (first_stage_y_[n]<0) {
      continue;
    } else {
      Vector temp;
      first_stage_x_[n].MakeColumnVector(first_stage_y_[n], &temp);
      term1+=la:;Dot(betas, temp) - log(exp_betas_x1_);
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
    if (first_stage_y_[n]>0) {
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
  postponed_probability.SetAll(0.0);
  for(index_t n=0; n<postponed_probability_.size(); n++) {
    for(index_t l=0; l<num_of_alphas_; l++) {
      double numerator=0;
      for(index_t i=0; i<second_stage_x_[n].n_cols(); i++) {
        nominator+=exp(la::Dot(betas.size(), betas.ptr(), ));
      }
      postponed_probability[n]+=numerator/(exp_betas_times_x1_+ numerator);
    }
    posponed_probability_*=alpha_wieght_;
  }
}

void ComputeExpBetasTimesX1_(Vector &betas) {
  exp_betas_times_x_1_=0;
  double sum=0;
  for(index_t i=0; i<first_stage_x_[n].n_cols(); i++) {
    exp_betas_times_x1_+=exp(la::Dot(betas.length(), 
                             beta.ptr(), 
                             first_stage_y_[n].GetColumnPtr(i)));
  }
}

// Weak learner

#include <algorithm>

#include "weak_learner.h"

DecisionStump::DecisionStump(T_IDX split_dim, T_IDX num_iter) 
  : sd_(split_dim), n_it_(num_iter) {
}
  
void DecisionStump::BatchLearn(Data *D) {
  vector<float> dval (D->Size(), 0.0);
  double intv, thd_test;
  T_IDX ct_g_pos, ct_g_neg, ct_l_pos, ct_l_neg;
  T_IDX ct_err, ct_err_new;
  T_IDX n_p_ex, n_n_ex; // number of positive/negative examples
  T_LBL gl_new;

  n_p_ex = 0; n_n_ex = 0;
  for (T_IDX x = 0; x < D->Size(); x++) {
    if (D->EXs_[x].y_ == 1)
      n_p_ex ++;
    else
      n_n_ex ++;
    for (T_IDX f=0; f<D->max_ft_idx_; f++) {
      if (D->EXs_[x].Fs_[f].i_ == sd_) {
        dval[x] = D->EXs_[x].Fs_[f].v_;
        break;
      }
      //else if (D->EXs_[x].Fs_[f].i_ > sd_) // need assume Fs to be sorted
      //  break;
    }
  }
  double max_dval = *max_element(dval.begin(), dval.end());
  double min_dval = *min_element(dval.begin(), dval.end());
  //cout << "sd: " << sd_ << "; max: " << max_dval << ", min: " << min_dval << endl;

  thd_ = min_dval;
  if (n_p_ex > n_n_ex) {
    gl_ = 1;
    ct_err = n_n_ex;
  }
  else {
    gl_ = -1;
    ct_err = n_p_ex;
  }
  if (max_dval == min_dval) {
    cout << "Warning! Decision stump with splitting dimension " 
         << sd_ << " is of constant values " << max_dval << " !" << endl;
  }
  else {
    intv = (max_dval - min_dval) / n_it_;
    thd_test = min_dval;
    for (T_IDX i=0; i<n_it_; i++) {
      thd_test += intv;
      ct_g_pos = 0; ct_g_neg = 0; ct_l_pos = 0; ct_l_neg = 0;
      for (T_IDX x=0; x<D->Size(); x++) {
        //ex = exs + x;
        if (dval[x] > thd_test) {
          if (D->EXs_[x].y_ == 1)
            ct_g_pos ++;
          else
            ct_g_neg ++;
        }
        else{
          if (D->EXs_[x].y_ == 1)
            ct_l_pos ++;
          else
            ct_l_neg ++;
        }
      }
      if ( (ct_g_pos + ct_l_neg) > (ct_l_pos + ct_g_neg) ) {
        gl_new = 1;
        ct_err_new = ct_l_pos + ct_g_neg;
      }
      else {
        gl_new = -1;
        ct_err_new = ct_g_pos + ct_l_neg;
      }
      //cout << "thd_test: " << thd_test << ", gl_new: " << gl_new << ", err_new: " << ct_err_new << endl;
      // better threshold found
      if (ct_err_new < ct_err) {
        thd_ = thd_test;
        gl_ = gl_new;
        ct_err = ct_err_new;
      }
    }
    //cout << "thd: " << thd << ", gl: " << gl << ", ct_err:" << ct_err << endl;
  }
}
  
T_LBL DecisionStump::PredictLabelBinary(Example *x) {
  double val = 0.0;
  for (T_IDX f = 0; f < x->Fs_.size(); f++) {
    if (x->Fs_[f].i_ == sd_) {
      val = x->Fs_[f].v_;
      break;
    }
  }
  if (val > thd_)
    return (T_LBL)gl_;
  else
    return (T_LBL)-gl_;
}


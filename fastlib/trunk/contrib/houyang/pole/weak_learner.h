#ifndef WEAKLEARNERS_H_
#define WEAKLEARNERS_H_

class WeakLearners {

public:
  // train a weak classifier
  virtual void WeakTrain(EXAMPLE *exs, size_t n_exs, double *weights) {
    return;
  };
  
  // prediction using weak learner
  virtual T_LBL WeakPredictLabel(EXAMPLE *ex) = 0;

  // Returns the name of weak learner
  virtual string GetName() const {
    return "";
  }
  
  virtual ~WeakLearners() {};
};


class DecisionStump : public WeakLearners {
 private:
  size_t n_iter;
  size_t sd; // splitting dimension
  float thd; // threshold for decision
  T_LBL gl; // label for > thd
 public:
  DecisionStump(size_t split_dim, size_t num_iter) {  
    sd = split_dim;
    n_iter = num_iter;
  }
  
  void WeakTrain(EXAMPLE *exs, size_t n_exs, double *weights) {
    vector<float> dval (n_exs, 0.0);
    float max_dval, min_dval, intv, thd_test;
    size_t ct_g_pos, ct_g_neg, ct_l_pos, ct_l_neg;
    size_t ct_err, ct_err_new;
    size_t n_exs_pos, n_exs_neg;
    T_LBL gl_new;
    EXAMPLE *ex = exs;
    n_exs_pos = 0; n_exs_neg = 0;
    for (size_t x=0; x<n_exs; x++) {
      ex = exs + x;
      if (ex->label == 1)
	n_exs_pos ++;
      else
	n_exs_neg ++;
      for (size_t f=0; f<ex->num_nz_feats; f++) {
	if (ex->feats[f].widx == sd) {
	  dval[x] = ex->feats[f].wval;
	  break;
	}
	//else if (ex->feats[f].widx > sd)
	//  break;
      }
    }
    max_dval = *max_element(dval.begin(), dval.end());
    min_dval = *min_element(dval.begin(), dval.end());
    //cout << "sd: " << sd << "; max: " << max_dval << ", min: " << min_dval << endl;

    thd = min_dval;
    if (n_exs_pos > n_exs_neg) {
      gl = 1;
      ct_err = n_exs_neg;
    }
    else {
      gl = -1;
      ct_err = n_exs_pos;
    }
    if (max_dval == min_dval) {
      cout << "Warning! Decision stump with splitting dimension " << sd << " is of constant values " << max_dval << " !" << endl;
    }
    else {
      intv = (max_dval - min_dval) / n_iter;
      thd_test = min_dval;
      for (size_t i=0; i<n_iter; i++) {
	thd_test += intv;
	ct_g_pos = 0; ct_g_neg = 0; ct_l_pos = 0; ct_l_neg = 0;
	for (size_t x=0; x<n_exs; x++) {
	  ex = exs + x;
	  if (dval[x] > thd_test) {
	    if (ex->label==1)
	      ct_g_pos ++;
	    else
	      ct_g_neg ++;
	  }
	  else{
	    if (ex->label==1)
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
	  thd = thd_test;
	  gl = gl_new;
	  ct_err = ct_err_new;
	}
      }
      //cout << "thd: " << thd << ", gl: " << gl << ", ct_err:" << ct_err << endl;
    }
    return;
  }
  
  T_LBL WeakPredictLabel(EXAMPLE *ex) {
    double val = 0.0;
    for (size_t f=0; f<ex->num_nz_feats; f++) {
      if (ex->feats[f].widx == sd) {
	val = ex->feats[f].wval;
	break;
      }
    }
    if (val > thd)
      return (T_LBL)gl;
    else
      return (T_LBL)-gl;
  }

  string GetName() const {
    return "stump";
  }
};


#endif


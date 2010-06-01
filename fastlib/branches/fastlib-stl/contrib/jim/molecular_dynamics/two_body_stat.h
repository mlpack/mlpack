#ifndef TWO_BODY_STAT_H
#define TWO_BODY_STAT_H

#include "fastlib/fastlib.h"

class TwoBodyStat{
 private:
 
  double coef_;
  int power_;

  FORBID_ACCIDENTAL_COPIES(TwoBodyStat);

 public:

  TwoBodyStat(){
  }

  ~TwoBodyStat(){
  }

  void Init(double coef_in, int power_in){       
    coef_ = coef_in;
    power_ = power_in;    
  } 

  void Init(const TwoBodyStat& left, const TwoBodyStat& right){   
    power_ = left.power_;
    coef_ = left.coef_ + right.coef_; 
  }

  const double coef(){
    return coef_;
  }

  const int power(){
    return power_;
  }

};

#endif

#ifndef THREE_BODY_STAT_H
#define THREE_BODY_STAT_H

#include "fastlib/fastlib.h"

class ThreeBodyStat{
 private:
 
  double coef_;
  int powA_, powB_, powC_;

  FORBID_ACCIDENTAL_COPIES(ThreeBodyStat);

 public:

  ThreeBodyStat(){
  }

  ~ThreeBodyStat(){
  }

  void Init(double coef_in, int powA_in, int powB_in, int powC_in){       
    coef_ = coef_in;
    powA_ = powA_in;   
    powB_ = powB_in;
    powC_ = powC_in;
  } 

  void Init(const ThreeBodyStat& left, const ThreeBodyStat& right){   
    powA_ = left.powA_;
    powB_ = left.powB_;
    powC_ = left.powC_;
    coef_ = left.coef_ + right.coef_;   
  }

  const double coef(){
    return coef_;
  }

  const int powA(){
    return powA_;
  }

  const int powB(){
    return powB_;
  }
  
  const int powC(){
    return powC_;
  }



};

#endif

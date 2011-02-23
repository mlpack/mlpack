#ifndef OPT_DWM_A_H
#define OPT_DWM_A_H

#include "learner.h"

class DWM_A : public Learner {
 public:
  DWM_A();
  void Learn();
  void Test();
};

#endif

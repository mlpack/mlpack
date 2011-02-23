#ifndef OPT_DWM_I_H
#define OPT_DWM_I_H

#include "learner.h"

class DWM_I : public Learner {
 public:
  DWM_I();
  void Learn();
  void Test();
};

#endif

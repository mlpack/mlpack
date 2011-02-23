#ifndef OPT_OEG_H
#define OPT_OEG_H

#include "learner.h"

class OEG : public Learner {
 public:
  OEG();
  void Learn();
  void Test();
};

#endif

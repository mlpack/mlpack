#ifndef OPT_OGD_H
#define OPT_OGD_H

#include "learner.h"

class OGD : public Learner {
 public:
  OGD();
  void Learn();
  void Test();
};

#endif

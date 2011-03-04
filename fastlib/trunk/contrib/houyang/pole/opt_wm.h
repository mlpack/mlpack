#ifndef OPT_WM_H
#define OPT_WM_H

#include "learner.h"

class WM : public Learner {
 public:
  WM();
  void Learn();
  void Test();
};

#endif

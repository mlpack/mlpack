#include "fastlib/fastlib.h"
#include "lin_alg.h"

int main(int argc, char *argv[]) {
  Matrix bob;
  bob.Init(2,2);
  bob.set(0,0,1);
  bob.set(0,1,2);
  bob.set(1,0,3);
  bob.set(1,1,4);

  Matrix sum;
  Sum(&bob, &sum);

  bob.PrintDebug("bob");
  sum.PrintDebug("sum");
  
  return SUCCESS_PASS;
}

#include <fastlib/fastlib.h>
#include <armadillo>
#include "tools.h"

using namespace arma;
using namespace std;


int main(int argc, char* argv[]) {
  mat X = randu(10,3);
  
  uvec rows_to_remove;
  rows_to_remove.set_size(3,1);
  
  
  rows_to_remove(0) = 1;
  rows_to_remove(1) = 3;
  rows_to_remove(2) = 5;
  
  mat X_mod;
  RemoveRows(X, rows_to_remove, X_mod);
  
  X.print("X");
  X_mod.print("X_mod");  
}

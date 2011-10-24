#include <fastlib/fastlib.h>
#include <armadillo>
#include "tools.h"

using namespace arma;
using namespace std;


void SpeedTest() {
  u32 k = 100;
  u32 n = 1000;
  
  mat X = randn(k, n);
  uvec rows_to_remove;
  u32 n_to_remove = 1;
  rows_to_remove.set_size(n_to_remove, 1);
  for(u32 i = 0; i < n_to_remove; i++) {
    rows_to_remove(i) = i;
  }


  //X.print("X");
  
  double n_secs_temp;
  double n_secs_mine = 0;
  double n_secs_theirs = 0;

  wall_clock timer;
  
  //start mine
  for(u32 t = 0; t < 1000; t++) {
    timer.tic();
    mat X_mod_mine;
    RemoveRows(X, rows_to_remove, X_mod_mine);
    n_secs_temp = timer.toc();
    n_secs_mine += n_secs_temp;
  }
  // end mine
  
  // start theirs
  for(u32 t = 0; t < 1000; t++) {
    mat X_mod_theirs = X;
    timer.tic();
    for(u32 i = 0; i < n_to_remove; i++) {
      X_mod_theirs.shed_row(rows_to_remove(i) - i);      
    }
    n_secs_temp = timer.toc();
    n_secs_theirs += n_secs_temp;
  }
  // end theirs
  
  //start mine
  for(u32 t = 0; t < 1000; t++) {
    timer.tic();
    mat X_mod_mine;
    RemoveRows(X, rows_to_remove, X_mod_mine);
    n_secs_temp = timer.toc();
    n_secs_mine += n_secs_temp;
  }
  n_secs_mine /= 2000.0;
  // end mine
  
  // start theirs
  for(u32 t = 0; t < 1000; t++) {
    mat X_mod_theirs = X;
    timer.tic();
    for(u32 i = 0; i < n_to_remove; i++) {
      X_mod_theirs.shed_row(rows_to_remove(i) - i);
    }
    n_secs_temp = timer.toc();
    n_secs_theirs += n_secs_temp;
  }
  n_secs_theirs /= 2000.0;
  // end theirs

  printf("mine:\t %e seconds\n", n_secs_mine);
  printf("theirs:\t %e seconds\n", n_secs_theirs);
}



int main(int argc, char* argv[]) {
  SpeedTest();
  return 1;

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

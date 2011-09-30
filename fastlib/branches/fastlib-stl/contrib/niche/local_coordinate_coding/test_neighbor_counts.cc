#include <fastlib/fastlib.h>
#include <armadillo>

using namespace arma;
using namespace std;


int main(int argc, char* argv[]) {
  u32 n_atoms_ = 5;
  u32 n_points_ = 10;
  
  mat V = randn(n_atoms_, n_points_);
  for(u32 i = 0; i < n_atoms_; i++) {
    for(u32 j = 0; j < n_points_; j++) {
      if(drand48() > 0.1) {
	V(i,j) = 0;
      }
    }
  }
  uvec adjacencies = find(V);
  
  V.print("V");
  adjacencies.print("adjacencies");
  
  
  // count number of atomic neighbors for each point x^i
  vec neighbor_counts = zeros(n_points_, 1);
  if(adjacencies.n_elem > 0) {
    // this gets the column index
    u32 cur_point_ind = (u32)(adjacencies(0) / n_atoms_);
    u32 cur_count = 1;
    for(u32 l = 1; l < adjacencies.n_elem; l++) {
      if((u32)(adjacencies(l) / n_atoms_) == cur_point_ind) {
	cur_count++;
      }
      else {
	neighbor_counts(cur_point_ind) = cur_count;
	cur_point_ind = (u32)(adjacencies(l) / n_atoms_);
	cur_count = 1;
      }
    }
    neighbor_counts(cur_point_ind) = cur_count;
  }
  
  
  neighbor_counts.print("neighbor counts");



}

#include <fastlib/fastlib.h>
#include <armadillo>

using namespace arma;
using namespace std;


int main(int argc, char* argv[]) {
  u32 n_atoms_ = 10;
  u32 n_points_ = 2;
  mat V_ = randn(n_atoms_, n_points_);
  
  // handle the case of inactive atoms (atoms not used in the given coding)
  std::vector<u32> inactive_atoms;
  std::vector<u32> active_atoms;
  active_atoms.reserve(n_atoms_);
  srand48(time(NULL));
  for(u32 j = 0; j < n_atoms_; j++) {
    if(drand48() > 0.5) {
      printf("inactive!\n");
      inactive_atoms.push_back(j);
    }
    else {
      active_atoms.push_back(j);
    }
  }
  
  printf("inactive atoms: ");
  for(u32 i = 0; i < inactive_atoms.size(); i++) {
    printf("%d ", inactive_atoms[i]);
  }
  printf("\n\n");

  printf("active atoms: ");
  for(u32 i = 0; i < active_atoms.size(); i++) {
    printf("%d ", active_atoms[i]);
  }
  printf("\n\n");

  mat active_V;
  if(inactive_atoms.empty()) {
    active_V = V_;
  }
  else {
    active_V.set_size(active_atoms.size(), n_points_);
    // first, check 0 to first inactive atom
    // now, check i'th inactive atom to (i + 1)'th inactive atom, until i = penultimate atom
    // now that i is last inactive atom, check last inactive atom to last atom
    
    // need to check this code

    u32 n_inactive_atoms = inactive_atoms.size();
    
    u32 cur_row = 0;
    u32 inactive_atom_ind = 0;
    if(inactive_atoms[0] > 0) {
      // note that this implies that n_atoms_ > 1
      u32 height = inactive_atoms[0];
      active_V(span(cur_row, cur_row + height - 1), span::all) =
	V_(span(0, inactive_atoms[0] - 1), span::all);
      cur_row += height;
    }
    while(inactive_atom_ind < n_inactive_atoms - 1) {
      u32 height = 
	inactive_atoms[inactive_atom_ind + 1]
	- inactive_atoms[inactive_atom_ind]
	- 1;
      if(height > 0) {
	active_V(span(cur_row, cur_row + height - 1), 
		 span::all) =
	  V_(span(inactive_atoms[inactive_atom_ind] + 1,
		  inactive_atoms[inactive_atom_ind + 1] - 1), 
	     span::all);
	cur_row += height;
      }
      inactive_atom_ind++;
    }
    if(inactive_atoms[inactive_atom_ind] < n_atoms_ - 1) {
      active_V(span(cur_row, active_atoms.size() - 1), 
	       span::all) = 
	V_(span(inactive_atoms[inactive_atom_ind] + 1, n_atoms_ - 1), 
	   span::all);
    }
  }
  
  V_.print("V");
  active_V.print("active_V");
}   
  

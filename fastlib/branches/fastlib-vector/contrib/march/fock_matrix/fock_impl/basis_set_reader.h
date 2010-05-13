#ifndef BASIS_SET_READER_H
#define BASIS_SET_READER_H 

/**
 * Reads basis set input files and a Matrix of atom centers and a list of atom
 * types.  Outputs a Matrix of basis centers, Vector of exponents, and Vector 
 * of momenta.  
 */
class BasisSetReader {
 
 public:

  void ReadBasisSet();
  
  void Init(const Matrix& atom_centers_in, 
            const ArrayList<char>& atom_types_in) {
  
    atom_centers.Copy(atom_centers_in);
    
    atom_types.InitCopy(atom_types_in, atom_types_in.capacity());
    
    num_atoms = atom_centers.n_cols();
  
  } // Init()

 private:

  Matrix atom_centers;
  
  ArrayList<char> atom_types;
  
  index_t num_atoms;
  

}; // class BasisSetReader




#endif
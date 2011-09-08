/*
 *  chem_reader.cc
 *  
 *
 *  Created by William March on 6/10/09.
 *
 */

#include "chem_reader.h"
#include "fastlib/fastlib.h"

namespace chem_reader {
  
  bool ReadQChemDensity(const char* filename, Matrix* mat, 
                             size_t mat_size) {
   
    bool retval = true;
    

    const char* delimiters = " ,";
    
    mat->Init(mat_size, mat_size);
    mat->SetAll(-999.99);
    
    size_t col_ind = 0;
    size_t master_col_ind = col_ind;
    
    TextLineReader reader;
    reader.Open(filename);
    
    // does this mean we have the first or second line?
    //reader.Gobble();
    
    while (master_col_ind < mat_size) {
      
      for (size_t row_ind = 0; row_ind < mat_size; row_ind++) {
      
        // not sure if I need this
        reader.Gobble();
        
        col_ind = master_col_ind;
        
        String this_row;
        this_row.Copy(reader.Peek());
        ArrayList<String> split;
        split.Init();
        
        this_row.Split(delimiters, &split);
        
        for (size_t i = 1; i < split.size(); i++) {
          
          // convert to double
          double new_val = atof(split[i]);
          
          // copy into matrix
          mat->set(row_ind, col_ind, new_val);
          col_ind++;
          
        } // for i
        
      } // for row_ind
      
      // skip the line with indices
      reader.Gobble();
      
      master_col_ind = col_ind;
        
    }
    

    
    return retval;
    
  }
  
}
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
  
  success_t ReadQChemDensity(const char* filename, Matrix* mat, 
                             index_t mat_size) {
   
    success_t retval = SUCCESS_PASS;
    

    const char* delimiters = " ,";
    
    mat->Init(mat_size, mat_size);
    mat->SetAll(-999.99);
    
    index_t col_ind = 0;
    index_t master_col_ind = col_ind;
    
    TextLineReader reader;
    reader.Open(filename);
    
    // does this mean we have the first or second line?
    //reader.Gobble();
    
    while (master_col_ind < mat_size) {
      
      for (index_t row_ind = 0; row_ind < mat_size; row_ind++) {
      
        // not sure if I need this
        reader.Gobble();
        
        col_ind = master_col_ind;
        
        String this_row;
        this_row.Copy(reader.Peek());
        ArrayList<String> split;
        split.Init();
        
        this_row.Split(delimiters, &split);
        
        for (index_t i = 1; i < split.size(); i++) {
          
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
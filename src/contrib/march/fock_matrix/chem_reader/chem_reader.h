/**
 *  chem_reader.h
 *  
 *
 *  Created by William March on 6/10/09.
 *
 */

#ifndef CHEM_READER_H
#define CHEM_READER_H

#include "fastlib/fastlib.h"

namespace chem_reader {
  
  //void ReadBasisSet();
  
  
  /**
   * Reads the density matrix in the format output by QChem.
   *
   * Assumes that the rest of the file has been stripped away.
   */
  bool ReadQChemDensity(const char* filename, Matrix* mat, 
                             size_t mat_size);
  
  
  
}


#endif
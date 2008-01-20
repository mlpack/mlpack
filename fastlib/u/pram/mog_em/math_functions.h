/**
 * @author pram
 * @file math_functions.h
 *
 */
#include "fastlib/fastlib.h"

/* Finds the index of the minimum element in each row of a matrix */

void min_element( Matrix& element, index_t *indices ){
	
  index_t last = element.n_cols() - 1;
  index_t first, lowest;
  index_t i;
	
  for( i = 0; i < element.n_rows(); i++ ){
		
    first = lowest = 0;
    if(first == last){
      indices[ i ] = last;
    }
    while(++first <= last){
      if( element.get( i , first ) < element.get( i , lowest ) ){
	lowest = first;
      }
    }
    indices[ i ] = lowest;
  }
  return;
}

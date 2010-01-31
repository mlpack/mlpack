/* MLPACK 0.2
 *
 * Copyright (c) 2008, 2009 Alexander Gray,
 *                          Garry Boyer,
 *                          Ryan Riegel,
 *                          Nikolaos Vasiloglou,
 *                          Dongryeol Lee,
 *                          Chip Mappus, 
 *                          Nishant Mehta,
 *                          Hua Ouyang,
 *                          Parikshit Ram,
 *                          Long Tran,
 *                          Wee Chin Wong
 *
 * Copyright (c) 2008, 2009 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file math_functions.h
 *
 * This file has certain functions that find the 
 * highest or lowest element in an array or 
 * in a row of a matrix and returns them
 *
 */
 
#include "fastlib/fastlib.h"
//#include "fastlib/fastlib_int.h"


/**
 * Finds the index of the minimum element
 * in each row of a matrix
 *
 * Example use:
 * @code
 * Matrix& mat;
 * index_t indices[mat.n_rows()];
 *
 * ...
 * 
 * min_element(mat, indices);
 * @endcode
 */

void min_element(Matrix& element, index_t *indices) {
	
	index_t last = element.n_cols() - 1;
	index_t first, lowest;
	index_t i;
	
	for (i = 0; i < element.n_rows(); i++) {
		
		first = lowest = 0;
		if (first == last) {
			indices[i] = last;
		}
		while (++first <= last) {
			if (element.get(i, first) < element.get(i, lowest)) {
				lowest = first;
			}
		}
		indices[i] = lowest;
	}
	return;
}

/**
 * Returns the index of the maximum element
 * in a float array 'array' of length 'length'.
 *
 * Example use:
 * @code
 * index_t length, index;
 * float array[length];
 * ...
 * index = max_element_index(array, length);
 * @endcode
 */
int max_element_index(float *array, int length) {
  
	int last = length - 1;
	int first = 0;
       	int highest = 0;
	
	if (first == last) {
		return last;
	}
	while (++first <= last) {
		if (array[first] > array[highest]) {
			highest = first;
		}
	}
	return highest;
}

/**
 * Finds the index of the maximum element in an arraylist
 * of floats
 *
 * Example use:
 * @code
 * index_t index;
 * ArrayList<float> array;
 * ...
 * index = max_element_index(array);
 * @endcode
 */
int max_element_index(ArrayList<float>& array){

        int last = array.size() - 1;
        int first = 0;
       	int highest = 0;

	if (first == last) {
                return last;
        }
        while (++first <= last) {
                if (array[first] > array[highest]) {
                        highest = first;
                }
	}
	return highest;
}

/**
 * Returns the index of the maximum element in 
 * an arraylist of doubles
 *
 * Example use:
 * @code
 * index_t index;
 * ArrayList<double> array;
 * ...
 * index = max_element_index(array);
 * @endcode
 */
int max_element_index(ArrayList<double>& array) {

        int last = array.size() - 1;
        int first = 0;
       	int highest = 0;

	if (first == last) {
                return last;
        }
        while (++first <= last) {
                if(array[first] > array[highest]) {
                        highest = first;
                }
	}
	return highest;
}


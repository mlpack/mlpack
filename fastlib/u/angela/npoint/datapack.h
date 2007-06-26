/**
 * @author: Angela N. Grigoroaia
 * @file: datapack.h
 *
 * @description: Read, store and manipulate coordinates and weights.
 */

/**
 * The main class is called 'DataPack' and it will contain a matrix that stores
 * all the relevant information about the given points (including their weights
 * if any are given). The first 'dimension' features represent the coordinates
 * and the following 'nweights' features are the weights of a particular point.
 */

/** 
 * Note:
 * Although everything is stored in a single matrix, the actual representation 
 * is abstracted by aliasing the coordinates and weights as two separate
 * matrices.
 */

#ifndef SIMPLE_DATA_H
#define SIMPLE_DATA_H

#include "fastlib/fastlib.h"
#include "globals.h"

class DataPack {
	private:
		Matrix data;
		int nweights;
		int dimension;
		int npoints;

	public:
		DataPack() {}
		~DataPack() {}

	public:
		/* Create an empty datapack. */
		void Init();

		/* Read the data from a file and specify the number of weights. */
		success_t InitFromFile(const char *file, const int weights);

		/* Modify the number of columns that are seen as weights. */
		void SetWeights(const int weights);

		/**
		 * Obtain aliases for the coordinates and weights of a given point. If the
		 * data is not initialized or there are no weights the functions will return
		 * SUCCESS_FAIL.
		 * This is the only method in which this class should interact with other
		 * parts of the code.
		 */
		success_t GetCoordinates(const index_t index, Vector &coordinates) const;
		success_t GetWeights(const index_t index, Vector &weights) const;

	/* Get useful values stored in the class */
	public:
		int num_weights() const {
			return nweights;
		}

		int num_dimensions() const {
			return dimension;
		}

		int num_points() const {
			return npoints;
		}	
};

#endif

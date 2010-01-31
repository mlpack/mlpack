/**
 * @author: Angela N. Grigoroaia
 * @file: datapack.cc
 */

#include "fastlib/fastlib.h"
#include "globals.h"

#include "datapack.h"

void DataPack::Init() {
	nweights = 0;
	dimension = 0;
	npoints = 0;
	/** 
	 * TODO ~> Cannot init data to NULL so I any reference to data will cause
	 * trouble. Should find a way to fix this. Maybe try something like:
	 * 		data.Init(0,0) ?
	 */
}


success_t DataPack::InitFromFile(const char *file, const int weights) {
	Dataset in;

	if ( !PASSED(in.InitFromFile(file)) ) {
		Init();
		fprintf(output,"Could not open '%s'. No datapoints available.\n",file);
		return SUCCESS_FAIL;
	}
	
	dimension = in.n_features() - weights;
	nweights = weights;
	npoints = in.n_points();
	data.Copy(in.matrix());
	return SUCCESS_PASS;
}


void DataPack::SetWeights(const int weights) {
	nweights = weights;
}


success_t DataPack::GetCoordinates(const index_t index, Vector &coordinates) 
	const {
	if (dimension < 1 || index < 0 || index > npoints) {
		return SUCCESS_FAIL;
	}
	data.MakeColumnSubvector(index,0,dimension, &coordinates);
	return SUCCESS_PASS;
}


success_t DataPack::GetWeights(const index_t index, Vector &weights)
	const {
	if (nweights < 0 || index < 0 || index > npoints) {
		return SUCCESS_FAIL;
	}
	data.MakeColumnSubvector(index,dimension,nweights, &weights);
	return SUCCESS_PASS;
}


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
	/** 
	 * TODO ~> Cannot init data to NULL so I any reference to data will cause
	 * trouble. Should find a way to fix this. Maybe try something like:
	 * 		data.Init(0,0) ?
	 */
}


success_t DataPack::InitFromFile(const char *file) {
	Dataset in;

	if ( !PASSED(in.InitFromFile(file)) ) {
		Init();
		fprintf(stderr,"Could not open '%s'. No datapoints available.\n",file);
		return SUCCESS_FAIL;
	}
	
	dimension = in.n_features();
	nweights = 0;
	data.Copy(in.matrix());
	return SUCCESS_PASS;
}


success_t DataPack::InitFromFile(const char *file, const int weights) {
	if( PASSED(InitFromFile(file)) ) {
		if (weights > 0) {
			SetWeights(weights);
			return SUCCESS_PASS;
		}
	}
	return SUCCESS_FAIL;
}


void DataPack::SetWeights(const int weights) {
	nweights = weights;
}


success_t DataPack::GetCoordinates(Matrix &coordinates) {	
	if(dimension <= 0) {
		return SUCCESS_FAIL;
	}

	coordinates.Alias(data.ptr(),data.n_rwos(),dimension);
	return SUCCESS_PASS;
}


success_t DataPack::GetWeights(Matrix &weights) {	
	index_t start_col = dimension;
	index_t n_cols = nweights;
		
	if (nweights <= 0) {
		return SUCCESS_FAIL;
	}

	data.MakeColumnSlice(start_col,n_cols,&weights);
	return SUCCESS_PASS;
}


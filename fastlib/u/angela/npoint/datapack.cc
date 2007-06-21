/**
 * @author: Angela N Grigoroaia
 * @date: 21.06.2007
 * @file: simple_data.cc
 */

void DataPack::Init() {
	nweights = 0;
	dimension = 0;
	data = NULL;
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
	data.Copy(in.Matrix());
	return SUCCESS_PASS
}


success_t DataPack::InitFromFile(const char *file, const char weights) {
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


Matrix DataPack::GetCoordinates() {
	if(dimension <= 0) {
		return NULL;
	}
}

Matrix DataPack::GetWeights() {
	if (nweights <=0) {
		return NULL;
	}
}

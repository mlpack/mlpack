#include "fastlib/fastlib.h"
#include "utils.h"


int main(int argc, char* argv[]) {

  const char* filename = "file.dat";

  ArrayList<GenMatrix<double> > data;

  LoadVaryingLengthData(filename,
			&data);

  for(int i = 0; i < data.size(); i++) {
    data[i].PrintDebug("");
  }
    



}

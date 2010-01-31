/*
 * =====================================================================================
 *
 *       Filename:  main.cc
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  11/21/2007 09:41:24 AM EST
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 *
 * =====================================================================================
 */

#include <sys/unistd.h>
#include <errno.h>
#include <string>
#include <vector>
#include "fastlib/fastlib.h"
#include "u/nvasil/spe/spe.h"

struct Parameters {
	std::string data_file_;
	std::string out_file_;
  std::string stress_file_;	
  index_t num_of_lambdas_;
	index_t newdim_;
	float32 lambda_max_;
	float32 lambda_min_;
	float32 range_;
	float32 tolerance_;
};
std::string Usage();
std::string PrintArgs(Parameters &args);
void PrintStress(std::vector<float32> &stress, std::string file);

int main(int argc, char *argv[]) {
	Parameters args;
	// initialize command line parameter
	fx_init(argc, argv);
	if (fx_param_exists(NULL, "help")) {
	  printf("%s\n", Usage().c_str());
		return -1;
	}
	NONFATAL("%s\n", Usage().c_str());
  args.data_file_=fx_param_str_req(NULL, "data_file");
  args.newdim_=fx_param_int(NULL, "newdim", 2);
  args.out_file_=fx_param_str(NULL, "out_file", "results");
	args.stress_file_=fx_param_str(NULL, "stress_file", "stress");
	args.num_of_lambdas_=fx_param_int(NULL, "n_lambdas", 100);
  args.lambda_max_=fx_param_double(NULL, "lambda_max", 2.0);
	args.lambda_min_=fx_param_double(NULL, "lambda_min",  0.1);
	args.range_=fx_param_double(NULL, "range", 0.1);
	args.tolerance_=fx_param_double(NULL, "tolerance", 0.1);
	NONFATAL("%s\n", PrintArgs(args).c_str());

  SPE spe;
	spe.Init(args.data_file_);
	spe.set_new_dimensions(args.newdim_);
	spe.set_lambdas(args.lambda_max_, args.lambda_min_, 
			            args.num_of_lambdas_);
	std::vector<float32> stress;
  spe.set_tolerance(args.tolerance_);
	spe.Optimize(args.range_, args.out_file_, stress);
	PrintStress(stress, args.stress_file_);
	spe.Destruct();
}

std::string Usage() {
  std::string ret=
		std::string("This program uses the stochastic proximity embedding SPE\n")+
		std::string("Parameters:\n")+
		std::string("--data_file=: text file containing the data,"
			          "	every column is a feature!!!!\n")+
		std::string("--out_file=: text file that contains the results, the same "
				        "format as the input file, (feature per column)\n")+
		std::string("--stress_file: this file contains the stress function " 
				        "per iteration\n")+
		std::string("--lambda_max: the maximum value for lambda\n")+
		std::string("--lambda_min: the minimum value for lambda\n")+
		std::string("--range: the range of the neighborhoods that the "
				        "algorithm operates\n")+
		std::string("--n_lambdas: the number of lambdas, or the number "
				        "of iterations\n")+
		std::string("--newdim: the embedded dimension (lower)\n")+
		std::string("--tolerance: the tolerance for terminating the algorithm\n")+
	  std::string("------------\n");
	  return ret;	
}

std::string PrintArgs(Parameters &args) {
  char temp[8192];
	sprintf(temp, "\ndata_file    : %s\n"
			          "out_file       : %s\n"
								"stress_file    : %s\n"
								"lambda_max     : %lg\n"
								"lambda_min     : %lg\n"
								"num_of_lambdas : %lli\n"
								"range          : %lg\n"
								"tolerance      : %lg\n",
								args.data_file_.c_str(),
								args.out_file_.c_str(),
								args.stress_file_.c_str(),
								(double)args.lambda_max_,
								(double)args.lambda_min_,
								(signed long long)args.num_of_lambdas_,
	              (double) args.range_,
								(double) args.tolerance_);
	return std::string(temp);							

}

void PrintStress(std::vector<float32> &stress, std::string file) {
  FILE *fp=NULL;
  if ((fp=fopen(file.c_str(), "w")) == NULL) {
    FATAL("Error: %s, while trying to open log file %s\n",
          strerror(errno), file.c_str());
  }
  for(index_t i=0; i < (index_t)stress.size() ; i++)	{
	  fprintf(fp, "%lg\n", (double)stress[i]);
	}
  if (fclose(fp)!=0) {
	  FATAL("Could not close %s, error %s encountered\n", 
			    file.c_str(), strerror(errno));
	}
}

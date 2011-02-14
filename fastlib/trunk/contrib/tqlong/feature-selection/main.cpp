/*
 * main.cpp
 *
 *  Created on: Feb 12, 2011
 *      Author: tqlong
 */

#include "dataset.h"
#include "l2lossl1reg.h"
#include <iostream>
#include <armadillo>
#include <boost/program_options.hpp>

using namespace std;
using namespace arma;
namespace po = boost::program_options;

void genData(const std::string& fileName);

int main(int argc, char** argv)
{
  string genFileName, dataFileName, weightFileName;
  int maxIter;
  double lambda, atol;

  cout << "Feature selection using sub-modular function inducing norms...\n";
  cout << "Solving optimization of the form: \\lambda \\Omega(w) + \\sum_i L(y_i, w'x_i)\n";

  po::variables_map vm;
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help", "produce this help message")
      ("gen", po::value<string>(&genFileName), "generate random data to file")
      ("data", po::value<string>(&dataFileName), "read labeled data from file")
      ("weight", po::value<string>(&weightFileName), "compute best weight vector to file")
      ("iter", po::value<int>(&maxIter)->default_value(100), "maximum number of iterations")
      ("lambda", po::value<double>(&lambda)->default_value(0.5), "regularization parameter")
      ("atol", po::value<double>(&lambda)->default_value(0.5), "regularization parameter")
      ;

  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    cout << desc << "\n";
    return 0;
  }

  if (vm.count("gen")) {
    genData(genFileName);
    return 0;
  }

  if (vm.count("data") && vm.count("weight")) {
    cout << "Computing best weight ...\n";
    DataSet data;
    data.load(dataFileName.c_str());
    cout << "n = " << data.n() << " dim = " << data.dim() << "\n";

    std::vector<double> params;
    params.push_back(lambda);
    params.push_back(maxIter);

    L2LossL1Reg<DataSet> algo(data);
    algo.setParameter(params);
    algo.run();
    algo.result().save(weightFileName.c_str());
    cout << "Result = \n" << algo.result() << "\n";
    return 0;
  }

  cout << desc << "\n";
  return 0;
}

void genData(const std::string& fileName)
{
  cout << "Generating random data ...\n";
  int n = 20, p = 2;
  mat X = arma::randn(n, p);
  vec w = arma::randn(p);
  w /= arma::norm(w,2);
  vec y = X*w;
  for (uint i = 0; i < y.n_elem; i++)
    y[i] = y[i] < 0 ? -1 : 1;
  DataSet(trans(X), y).save(fileName.c_str());
  w.save((fileName+"-truth").c_str(), arma_ascii);
}

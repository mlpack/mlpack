/*
 * dataset.cpp
 *
 *  Created on: Feb 14, 2011
 *      Author: tqlong
 */

#include "dataset.h"
#include <fstream>
#include <cassert>

DataSet::DataSet(const arma::mat &X, const arma::vec &y)
  : X_(X), y_(y)
{
}

DataSet::~DataSet() {
  // TODO Auto-generated destructor stub
}

void DataSet::load(const char* fileName)
{
  std::ifstream f;
  f.open(fileName);
  load(f);
  f.close();
}

void DataSet::load(std::istream& stream)
{
  X_.load(stream);
  X_ = trans(X_);
  char s[1000];
  stream.getline(s, 1000);
  y_.load(stream);
//  std::cout << "n = " << n() << " dim = " << dim() << "\n";
//  std::cout << "y.n = " << y_.n_elem << "\n";
  assert(X_.n_cols == y_.n_elem);
}

void DataSet::save(const char* fileName, arma::file_type type)
{
  std::ofstream f;
  f.open(fileName);
  save(f, type);
  f.close();
}

void DataSet::save(std::ostream& stream, arma::file_type type)
{
  arma::mat(arma::trans(X_)).save(stream, type); // saving column-wisely for easy reading
  y_.save(stream, type);
}

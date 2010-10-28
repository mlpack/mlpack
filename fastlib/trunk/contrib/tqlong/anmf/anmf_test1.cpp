#include <fastlib/fastlib.h>
#include <iostream>
#include <boost/program_options.hpp>
#include "anmf.h"

//namespace po = boost::program_options;
using namespace std;

boost::program_options::options_description desc("Allowed options");
boost::program_options::variables_map vm;

void process_options(int argc, char** argv)
{
  desc.add_options()
      ("help", "produce help message")
      ("reference", boost::program_options::value<string>()->default_value("reference.txt") ,"file consists of reference points")
      ("query", boost::program_options::value<string>()->default_value("query.txt") ,"file consists of query points")
      ("matrix", boost::program_options::value<string>()->default_value("matrix.txt") ,"file consists of weight matrix")
      ("random", boost::program_options::value<int>()->default_value(0) ,"generate random reference and query points");

  boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
  boost::program_options::notify(vm);

  if (vm.count("help"))
  {
    cout << desc << endl;
    exit(1);
  }
}

class DistanceMatrix
{
  const Matrix &reference_, &query_;
  std::vector<double> price_;
public:
  DistanceMatrix(const Matrix &reference, const Matrix &query)
    : reference_(reference), query_(query), price_(query.n_cols(), 0)
  {
    DEBUG_ASSERT(reference.n_rows() == query.n_rows());
  }
  int n_rows() const { return reference_.n_cols(); }
  int n_cols() const { return query_.n_cols(); }
  double get(int i, int j) const
  {
    Vector r_i, q_j;
    reference_.MakeColumnVector(i, &r_i);
    query_.MakeColumnVector(j, &q_j);
    return -sqrt(la::DistanceSqEuclidean(r_i, q_j));
  }
  void setPrice(int j, double price)
  {
    price_[j] = price;
  }
  void getBestAndSecondBest(int bidder, int &best_item, double &best_surplus, double &second_surplus)
  {
    best_surplus = second_surplus = -std::numeric_limits<double>::infinity();
    for (int item = 0; item < query_.n_cols(); item++)
    {
      double surplus = get(bidder, item) - price_[item];
      if (surplus > best_surplus)
      {
        best_item = item;
        second_surplus = best_surplus;
        best_surplus = surplus;
      }
      else if (surplus > second_surplus)
      {
        second_surplus = surplus;
      }
    }
  }
};

void generateRandom(const char* refFile, const char* queryFile)
{
  int n_points = vm["random"].as<int>();
  int dim = 2;
  Matrix ref, query;
  ref.Init(dim, n_points);
  query.Init(dim, n_points);

  for (int i = 0; i < dim; i++)
  {
    for (int j = 0; j < n_points; j++)
    {
      ref.ref(i, j) = math::Random(0, 10);
      query.ref(i, j) = math::Random(0, 10);
    }
  }
  data::Save(refFile, ref);
  data::Save(queryFile, query);
}

int main(int argc, char** argv)
{
  process_options(argc, argv);

  if (vm["random"].as<int>() > 0)
  {
    generateRandom(vm["reference"].as<string>().c_str(), vm["query"].as<string>().c_str());
  }

  if (vm.count("reference") && vm.count("query"))
  {
    Matrix ref, query;
    data::Load(vm["reference"].as<string>().c_str(), &ref);
    data::Load(vm["query"].as<string>().c_str(), &query);

    DistanceMatrix M(ref, query);
    anmf::AuctionMaxWeightMatching<DistanceMatrix> matcher(M, true);
    for (int refIndex = 0; refIndex < ref.n_cols(); refIndex++)
      cout << "reference point " << refIndex << " --> query point " << matcher.leftMatch(refIndex) << endl;

    return 0;
  }

//  if (vm.count("matrix"))
//  {
//    Matrix M;
//    data::Load(vm["matrix"].as<string>().c_str(), &M);

//    anmf::AuctionMaxWeightMatching<Matrix> matcher(M, true);
//    for (int bidder = 0; bidder < M.n_rows(); bidder++)
//      cout << "bidder " << bidder << " --> " << matcher.leftMatch(bidder) << endl;
//    return 0;
//  }

  return 0;
}

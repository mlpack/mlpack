#include <fastlib/fastlib.h>
#include <iostream>
#include <boost/program_options.hpp>
#include "anmf.h"

namespace po = boost::program_options;
using namespace std;

boost::program_options::options_description desc("Allowed options");
boost::program_options::variables_map vm;

void process_options(int argc, char** argv)
{
  desc.add_options()
      ("help", "produce help message")
      ("reference", po::value<string>()->default_value("reference.txt") ,"file consists of reference points")
      ("query", po::value<string>()->default_value("query.txt") ,"file consists of query points")
      ("matrix", po::value<string>()->default_value("matrix.txt") ,"file consists of weight matrix");

  boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
  boost::program_options::notify(vm);

  if (vm.count("help"))
  {
    cout << desc << endl;
    exit(1);
  }
}

int main(int argc, char** argv)
{
  process_options(argc, argv);

  Matrix M;

  if (vm.count("matrix"))
  {
    data::Load(vm["matrix"].as<string>().c_str(), &M);
//    ot::Print(M);
  }

  anmf::AuctionMaxWeightMatching<Matrix> matcher(M);

  return 0;
}

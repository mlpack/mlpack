/* Parallel Online Learning Experiments
 *
 * Build: cmake ..make
 * Build: g++ -lboost_program_options -pthread pole_pt.cc -o pole_pt
 * Build: g++ -I/scratch/app/boost/ -L/scratch/app/boost/lib -lboost_program_options -pthread pole_pt.cc -o pole_pt
 * Usage: ./pole_pt -d ../heart_scale -m ogd --random -calc_loss --bias --reg_factor 0.01 -e 30 -t 8
 *        ./pole_pt -d ../ijcnn1 --random -m ogd --calc_loss --bias --comm 1 -c 1 -e 1 -t 1
 *        ./pole_pt -d ../ijcnn1 -m ogd --calc_loss --comm 1 -c 1 -t 1 --bias --random -b 100 -e 100
 *        ./pole_pt -d ../svmguide1 -m ogd --calc_loss --comm 1 -c 1 -t 1 -b 1 --bias -e 3000
 *
 */

#include "pole.h"

using namespace boost::posix_time;

int main(int argc, char *argv[]) {
  // Use boost's program_options.
  boost_po::options_description desc("POLE (Parallel Online Learning Experiments) options");

  // Parse input parameters, read data.
  boost_po::variables_map vm = ParseArgs(argc, argv, l1, desc);

  used_ct = 0;
  epoch_ct = 0;
  iter_res_ct = 0;

  if (global.random_input) {
    srand(time(NULL));
  }

  ReadData(vm);

  // Begin training process
  ptime time_start(microsec_clock::local_time());
  if (global.opt_method == "ogd") {
    cout << "Using Online subGradient Descent (OGD)..." << endl;
    Ogd(l1);
  }
  else if (global.opt_method == "d_sgd") {
    cout << "Using Delayed Stochastic subGradient Descent ..." << endl;
    //train_delayed_sgd(l1);
  }
  else if (global.opt_method == "d_dcd") {
    cout << "Using Delayed Dual Coordinate Descent..." << endl;
    //train_delayed_dcd(l1);
  }
  else {
    cout << "Unknown optimiztion method. Using default: Online subGradient Descent (OGD)..." << endl;
    Ogd(l1);
  }
  
  ptime time_end(microsec_clock::local_time());
  time_duration duration(time_end - time_start);
  cout << "Duration: " << duration << endl;
  
}

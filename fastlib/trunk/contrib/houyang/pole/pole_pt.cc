/* Parallel Online Learning Experiments
 *
 * Build: cmake .. make
 * Usage: ./pole_pt -d ../heart_scale --random 0 -m ogd --type classification -l hinge --calc_loss --bias 1 -c 1 -e 3000 --threads 8
 *        ./pole_pt -d ../ijcnn1 -m ogd --calc_loss --comm 1 -c 1 --bias 1 -r 2 --random -e 10 --threads 100 -b 1
 *        ./pole_pt -d ../svmguide1 -m ogd --calc_loss --comm 1 -c 0.1 --threads 1 -b 1 --bias 1 -e 3000
 *        ./pole_pt -d ../svmguide1 -m oeg --type classification -l hinge --calc_loss --bias 1 --comm 1 -c 1000 -i 5000000 --threads 1 -b 1
 *        ./pole_pt -d ../rcv1_train -m ogd --type classification -l hinge --calc_loss --comm 1 -c 1 -e 2 --threads 1 -b 1
 *        ./pole_pt -d ../rcv1_train -m oeg --type classification -l hinge --calc_loss --comm 1 -c 100 -i 5000 --threads 1 -b 1
 *        ./pole_pt -d ../cod-rna -m dwm_a --type classification --calc_loss 1 --comm 1 -b 1 --threads 1 -a 0.9 -p 8 -i 6000
 *
 */

#include "pole.h"

int main(int argc, char *argv[]) {
  Pole p;
  // Parse input arguments
  p.ParseArgs(argc, argv);

  ptime time_start(microsec_clock::local_time());
  // Learning
  p.Run();
  ptime time_end(microsec_clock::local_time());
  time_duration duration(time_end - time_start);
  
  cout << "--------------------------------------------" << endl;
  cout << "Total time: " << duration << endl;
  cout << "Total time in ms: " << duration.total_milliseconds() << endl;

}

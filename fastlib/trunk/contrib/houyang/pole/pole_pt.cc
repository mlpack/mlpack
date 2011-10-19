/* Parallel Online Learning Experiments
 *
 * Build: cmake .. make
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
  cout << "Total time in ms: " << duration.total_milliseconds() << endl << endl;

}

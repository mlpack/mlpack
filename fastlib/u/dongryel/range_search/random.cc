//  Generates some random numbers and writes them to the screen
#include <iostream>
#include <ios>
#include <iomanip>
#include <fstream>
#include <math.h>

int main ()
{
  using std::string;
  using std::cout;
  using std::cin;
  using std::endl;

  // Ask user to specify how many numbers to generate
//   cout << "Please enter the number of random numbers to generate: ";
//   int ngen;
//   cin >> ngen;

  int ngen=1000000;

  // Print out the largest random number that the system can generate.
  //  cout << RAND_MAX << endl << endl;

  // Loop according to the number of times the user has chosen and generate
  // the random numbers
  for (int counter=0; counter!=ngen ; ++counter) {
    double myRan0 = (double) rand() / (double) RAND_MAX;
    double myRan1 = myRan0 + 0.1*((double) rand() / (double) RAND_MAX);
    double myRan2 = myRan0 + 0.1*((double) rand() / (double) RAND_MAX);
    double myRan3 = myRan0 + 0.1*((double) rand() / (double) RAND_MAX);

    cout<< myRan0 << ", " 
	<< myRan1 << ", "
	<< myRan2 << ", "
	<< myRan3 << endl;
  }
  return 0;
}

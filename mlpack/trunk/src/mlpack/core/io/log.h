#ifndef MLPACK_CLI_LOG_H
#define MLPACK_CLI_LOG_H

#include "prefixedoutstream.h"
#include "nulloutstream.h"



namespace mlpack {

/*
 * This class focuses on facilitating formatted output to the terminal.
 */
class Log {
 public:  
  /*
   * Checks if the specified condition is true.
   * If not, halts program execution and prints a custom error message.
   * Does nothing in non-debug mode.
   */
  static void Assert(bool condition, 
      const char* message="Assert Failed.");


  // We only use PrefixedOutStream if the program is compiled with debug
  // symbols
#ifdef DEBUG
  // Prints debug output with the appropriate tag.
  static io::PrefixedOutStream Debug;
#else
  // Dumps debug output into the bit nether regions.
  static io::NullOutStream Debug;
#endif
  // Prints output with their respective tags of [INFO], [WARN], and [FATAL]
  static io::PrefixedOutStream Info;
  static io::PrefixedOutStream Warn;
  static io::PrefixedOutStream Fatal;
  static std::ostream& cout;
};

}; //namespace mlpack
#endif //mlpack_io_log_h

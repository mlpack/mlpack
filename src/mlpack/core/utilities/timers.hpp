#ifndef MLPACK_TIMERS_H
#define MLPACK_TIMERS_H
 
#include <map>
#include <string>

#ifndef _WIN32
  #include <sys/time.h> //linux
#else
  #include <winsock.h> //timeval on windows
  #include <windows.h> //GetSystemTimeAsFileTime on windows
//gettimeofday has no equivalent will need to write extra code for that.
  #if defined(_MSC_VER) || defined(_MSC_EXTENSCLINS)
    #define DELTA_EPOCH_IN_MICROSECS 11644473600000000Ui64
  #else
    #define DELTA_EPOCH_IN_MICROSECS 11644473600000000ULL
  #endif
#endif //_WIN32

namespace mlpack {

class Timers {
 public:
 /*
  * Returns a copy of all the timers used via this interface.
  */
  static std::map<std::string, timeval> GetAllTimers();

 /* 
  * Returns a copy of the timer specified.
  *
  * @param timerName The name of the timer in question.
  */
  static timeval GetTimer(const char* timerName);

 /*
  * Prints the specified timer.  If it took longer than a minute to complete
  * the timer will be displayed in days, hours, and minutes as well.
  *
  * @param timerName The name of the timer in question.
  */
  static void PrintTimer(const char* timerName);

 /*
  * Initializes a timer, available like a normal value specified on
  * the command line.  Timers are of type timval
  *
  * @param timerName The name of the timer in question.
  */
  static void StartTimer(const char* timerName);

 /*
  * Halts the timer, and replaces it's value with
  * the delta time from it's start
  *
  * @param timerName The name of the timer in question.
  */
  static void StopTimer(const char* timerName);
 private:
  static std::map<std::string, timeval> timers;

  void FileTimeToTimeVal(timeval* tv);

  //Don't want any instances floating around.
  Timers();
  ~Timers();
};

}; //namespace mlpack

#endif //MLPACK_TIMERS_H

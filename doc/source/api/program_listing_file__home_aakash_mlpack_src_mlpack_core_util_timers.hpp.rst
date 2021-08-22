
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_util_timers.hpp:

Program Listing for File timers.hpp
===================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_util_timers.hpp>` (``/home/aakash/mlpack/src/mlpack/core/util/timers.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_UTILITIES_TIMERS_HPP
   #define MLPACK_CORE_UTILITIES_TIMERS_HPP
   
   #include <atomic>
   #include <chrono> // chrono library for cross platform timer calculation.
   #include <iomanip>
   #include <list>
   #include <map>
   #include <mutex>
   #include <string>
   #include <thread> // std::thread is used for thread safety.
   
   #if defined(_WIN32)
     // uint64_t isn't defined on every windows.
     #if !defined(HAVE_UINT64_T)
       #if SIZEOF_UNSIGNED_LONG == 8
         typedef unsigned long uint64_t;
       #else
         typedef unsigned long long uint64_t;
       #endif // SIZEOF_UNSIGNED_LONG
     #endif // HAVE_UINT64_T
   #endif
   
   namespace mlpack {
   
   class Timer
   {
    public:
     static void Start(const std::string& name);
   
     static void Stop(const std::string& name);
   
     static std::chrono::microseconds Get(const std::string& name);
   
     static void EnableTiming();
   
     static void DisableTiming();
   
     static void ResetAll();
   
     static std::map<std::string, std::chrono::microseconds> GetAllTimers();
   };
   
   namespace util {
   
   class Timers
   {
    public:
     Timers() : enabled(false) { }
   
     std::map<std::string, std::chrono::microseconds> GetAllTimers();
   
     void Reset();
   
     std::chrono::microseconds Get(const std::string& timerName);
   
     static std::string Print(const std::chrono::microseconds& totalDuration);
   
     void Start(const std::string& timerName,
                const std::thread::id& threadId = std::thread::id());
   
     void Stop(const std::string& timerName,
               const std::thread::id& threadId = std::thread::id());
   
     void StopAllTimers();
   
     std::atomic<bool>& Enabled() { return enabled; }
     bool Enabled() const { return enabled; }
   
    private:
     std::map<std::string, std::chrono::microseconds> timers;
     std::mutex timersMutex;
     std::map<std::thread::id, std::map<std::string,
         std::chrono::high_resolution_clock::time_point>> timerStartTime;
   
     std::atomic<bool> enabled;
   };
   
   } // namespace util
   } // namespace mlpack
   
   #endif // MLPACK_CORE_UTILITIES_TIMERS_HPP

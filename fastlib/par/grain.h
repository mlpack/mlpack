// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file grain.h
 *
 * Tool for creating very simple parallel programs by enqueuing small work
 * items in a priority queue and executing them greedily.
 *
 * @removal
 */

#ifndef PAR_GRAIN_H
#define PAR_GRAIN_H

#include "col/heap.h"

#include "thread.h"

/**
 * Simple difficulty-based work queue for easy parallelization.
 *
 * To use this, simply divide up your work into grains (as shown below) that
 * are probably a bit smaller than what one thread should be able to handle. 
 * Enqueue each grain into the GrainQueue, associating it with a
 * "difficulty" measure which should estimate the relative amount of time
 * for each grain.  You can then use this to automatically run a number of
 * threads.
 *
 * To allow grains to be sent over multiple machines, put only basic data
 * into your grains, and associate the grain queue with a context, which
 * is a pointer to something.  This context will be passed to every grain
 * when it is run.
 *
 * TODO: This has a bug that, if any of the threads find an empty queue,
 * that thread will die.  This would prohibit you from recursively building
 * a kd-tree or such, as after the root node is de-queued all the other
 * threads would die because they think there is no work to do.
 *
 * This class is thread safe, so it is perfectly fine for grains to put
 * more work on the grain queue.
 *
 * TODO: Update documentation to reflect the fact that ThreadedGrainRunner
 * is a separate class now.
 *
 * @code
 * struct SolverGrain {
 *   Solver *solver;
 *   int a;
 *   int b;
 *   ~SolverGrain() {}
 *   SolverGrain(Solver *solver_in, int a_in, int b_in) {
 *     solver = solver_in;
 *     a = a_in;
 *     b = b_in;
 *   }
 *   void Run() {
 *     solver->Solve(a, b);
 *   }
 * };
 *
 * class Solver {
 *   void SolveRange(int a_min, int a_max, int b_min, int b_max) {
 *     GrainQueue&lt;SolverGrain&gt; queue;
 *     for (int a = 0; a < 10; a++) {
 *       for (int b = 0; b < 10; b++) {
 *         queue->Put(a * b, new SolverGrain(this, a, b));
 *       }
 *     }
 *   }
 *   void Solve(int a, int b) {....}
 * }
 * @endcode
 */
template<typename TGrain>
class GrainQueue {
 public:
  typedef TGrain Grain;

 private:
  MinHeap<double, Grain*> queue_;
  Mutex mutex_;
  
 public:
  GrainQueue() {}
  ~GrainQueue() {}
  
  /**
   * Initializes.
   */
  void Init() {
    queue_.Init();
  }
  
  /**
   * Puts a grain into the queue to be dispatched.
   *
   * @param difficulty relative problem difficulty
   */
  void Put(double difficulty, Grain *grain) {
    mutex_.Lock();
    queue_.Put(-difficulty, grain);
    mutex_.Unlock();
  }
  
  /**
   * Pops the most desirable grain to work on.
   *
   * You might not have to call this yourself.
   */
  Grain *Pop() {
    mutex_.Lock();
    Grain *result = likely(queue_.size() != 0) ? queue_.Pop() : NULL;
    mutex_.Unlock();
    return result;
  }
  
  /**
   * Gets the size of this queue.
   */
  index_t size() const {
    return queue_.size();
  }
};

template<typename TGrain, typename TContext = int>
class ThreadedGrainRunner {
  FORBID_COPY(ThreadedGrainRunner);
 public:
  typedef TGrain Grain;
  typedef TContext Context;

 private:
  struct ThreadTask : public Task {
    ThreadedGrainRunner *runner_;
    
    ThreadTask(ThreadedGrainRunner *runner_in) {
      runner_ = runner_in;
    }
    
    void Run() {
      while (runner_->RunOneGrain()) {}
      delete this;
    }
  };
  
 private:
  GrainQueue<Grain> *queue_;
  Context context_;
  
 public:
  ThreadedGrainRunner() {}
  ~ThreadedGrainRunner() {}
  
  void Init(GrainQueue<Grain> *queue_in, Context context_in) {
    queue_ = queue_in;
    context_ = context_in;
  }
    
  /**
   * Pops and runs one task.
   *
   * Use this if you, for some reason, decided that running separate threads
   * was a bad idea and you really just want to run grains yourself.
   *
   * @return true whether a task was run, false if no more tasks left
   */
  bool RunOneGrain() {
    Grain *grain = queue_->Pop();
    if (unlikely(!grain)) {
      return false;
    } else {
      grain->Run(context_);
      delete grain;
      return true;
    }
  }
  
  /**
   * Spawns a single running thread.
   *
   * You must WaitStop or Detach this thread, and eventually, free
   * the returned object.
   *
   * (TODO: In the future you might have to delete its task() too).
   *
   * @return a newly created thread
   */
  Thread *SpawnThread() {
    ThreadTask *task = new ThreadTask(this);
    Thread *thread = new Thread();
    thread->Init(task);
    thread->Start();
    return thread;
  }

  /**
   * Creates the specified number of threads, and uses those to execute
   * all grains of work.
   */
  void RunThreads(int num_threads) {
    ArrayList<Thread*> threads;
    
    threads.Init(num_threads);
    
    for (int i = 0; i < num_threads; i++) {
      threads[i] = SpawnThread();
    }
    for (int i = 0; i < num_threads; i++) {
      threads[i]->WaitStop();
      delete threads[i];
    }
  }
};

#endif

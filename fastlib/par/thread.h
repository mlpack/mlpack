// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file thread.h
 *
 * Abstractions for helping you write threaded programs.
 */
#ifndef PAR_THREAD_H
#define PAR_THREAD_H

#include "task.h"

#include "base/common.h"

#include <pthread.h>

/**
 * Thread convenience wrapper.
 *
 * Usage: Create a Thread, give it a Task object, tell the thread to run,
 * and eventually wait for the thread to finish.
 */
class Thread {
  FORBID_COPY(Thread);
  
 private:
#ifdef DEBUG
  enum {UNINIT, READY, ATTACHED, DETACHED, DONE} status_;
#endif
  pthread_t thread_;
  Task *task_;
  
  static void *ThreadMain_(void *self) {
    Thread* thread = reinterpret_cast<Thread*>(self);
    
    thread->task_->Run();
    
    return NULL;
  }

  void Exit_() {
    pthread_exit(NULL);
  }
  
 public:
  Thread() {
    DEBUG_ONLY(status_ = UNINIT);
  }
  ~Thread() {
    DEBUG_ASSERT(status_ == DETACHED || status_ == READY || status_ == DONE);
    DEBUG_ONLY(status_ = UNINIT);
  }
  
  /**
   * Initializes, given a task to run.
   */
  void Init(Task* task_in) {
    DEBUG_ASSERT(status_ == UNINIT);
    task_ = task_in;
    DEBUG_ONLY(status_ = READY);
  }
  
  /**
   * Starts the thread running.
   */
  void Start() {
    DEBUG_ASSERT(status_ == READY);
    pthread_create(&thread_, NULL,
        ThreadMain_, reinterpret_cast<void*>(this));
    DEBUG_ONLY(status_ = ATTACHED);
  }
  
  /**
   * Detaches a thread -- the thread will cease to exist once the task
   * completes.  You may not call WaitStop on this thread afterwards.
   */
  void Detach() {
    DEBUG_ASSERT(status_ == ATTACHED);
    pthread_detach(thread_);
    DEBUG_ONLY(status_ = DETACHED);
  }
  
  /**
   * Wait for a thread to stop.
   *
   * Failure to do this may cause your program to hang when it is done.
   */
  void WaitStop() {
    DEBUG_ASSERT(status_ == ATTACHED);
    pthread_join(thread_, NULL);
    DEBUG_ONLY(status_ = DONE);
  }
  
  /**
   * Gets the contained task.
   */
  Task* task() const {
    return task_;
  }
};

/**
 * Mutual exclusion lock to prevent threads from clobbering results.
 */
class Mutex {
  FORBID_COPY(Mutex);
  friend class WaitCondition;
  
 private:
  pthread_mutex_t mutex_;
 
 public:
  static Mutex global;
 
 public:
  Mutex() {
    pthread_mutex_init(&mutex_, NULL);
  }
  
  ~Mutex() {
    pthread_mutex_destroy(&mutex_);
  }
  
  /** Obtains the lock. */
  void Lock() {
    pthread_mutex_lock(&mutex_);
  }
  
  /** Tries to lock, returns false if doing so would require waiting. */
  bool TryLock() {
    return likely(!pthread_mutex_trylock(&mutex_));
  }
  
  /** Releases the lock. */
  void Unlock() {
    pthread_mutex_unlock(&mutex_);
  }
};

/**
 * Wait condition for alerting other threads of an action.
 */
class WaitCondition {
  FORBID_COPY(WaitCondition);
  
 private:
  pthread_cond_t cond_;
 
 public:
  WaitCondition() {
    pthread_cond_init(&cond_, NULL);
  }
  
  ~WaitCondition() {
    pthread_cond_destroy(&cond_);
  }
  
  void Signal() {
    pthread_cond_signal(&cond_);
  }
  
  void Broadcast() {
    pthread_cond_broadcast(&cond_);
  }
  
  void Wait(Mutex* mutex_to_unlock) {
    pthread_cond_wait(&cond_, &mutex_to_unlock->mutex_);
  }
  
  void WaitMillis(Mutex& mutex_to_unlock, unsigned millis) {
    struct timespec ts;
    
    ts.tv_sec = millis / 1000;
    ts.tv_nsec = (millis % 1000) * 1000000;
    
    pthread_cond_timedwait(&cond_, &mutex_to_unlock.mutex_, &ts);
  }
  
  void WaitSec(Mutex& mutex_to_unlock, unsigned sec) {
    struct timespec ts;
    
    ts.tv_sec = sec;
    ts.tv_nsec = 0;
    
    pthread_cond_timedwait(&cond_, &mutex_to_unlock.mutex_, &ts);
  }
};

/**
 * Mix-in to make a version of an existing object that can be locked.
 *
 * Your object must have default constructors and use Init methods.
 * The resulting object will have Lock, Unlock, and TryLock methods.
 */
template<class TContained>
class Lockable : public TContained, public Mutex {
  FORBID_COPY(Lockable);
  
  Lockable() {}
  ~Lockable() {}
};

#endif

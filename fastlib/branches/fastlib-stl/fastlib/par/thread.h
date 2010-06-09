/* MLPACK 0.2
 *
 * Copyright (c) 2008, 2009 Alexander Gray,
 *                          Garry Boyer,
 *                          Ryan Riegel,
 *                          Nikolaos Vasiloglou,
 *                          Dongryeol Lee,
 *                          Chip Mappus, 
 *                          Nishant Mehta,
 *                          Hua Ouyang,
 *                          Parikshit Ram,
 *                          Long Tran,
 *                          Wee Chin Wong
 *
 * Copyright (c) 2008, 2009 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
/**
 * @file thread.h
 *
 * Abstractions for helping you write threaded programs.
 */
#ifndef PAR_THREAD_H
#define PAR_THREAD_H

#include "task.h"

#include "../base/base.h"

#include <pthread.h>

/**
 * Thread convenience wrapper.
 *
 * Usage: Create a Thread, give it a Task object, tell the thread to run,
 * and eventually wait for the thread to finish.
 */
class Thread {
  FORBID_ACCIDENTAL_COPIES(Thread);
  
 public:
  enum {
    LOW_PRIORITY = 20,
    NORMAL_PRIORITY = 0
  };
  
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
#ifdef DEBUG
    DEBUG_ONLY(status_ = UNINIT);
#endif
  }
  ~Thread() {
#ifdef DEBUG
    DEBUG_ASSERT(status_ == DETACHED || status_ == READY || status_ == DONE || status_ == UNINIT);
    DEBUG_ONLY(status_ = UNINIT);
#endif
  }
  
  /**
   * Initializes, given a task to run.
   */
  void Init(Task* task_in) {
#ifdef DEBUG
    DEBUG_ASSERT(status_ == UNINIT);
#endif
    task_ = task_in;
#ifdef DEBUG
    DEBUG_ONLY(status_ = READY);
#endif
  }
  
  /**
   * Starts the thread running.
   */
  void Start() {
#ifdef DEBUG
    DEBUG_ASSERT(status_ == READY);
#endif
    pthread_create(&thread_, NULL,
        ThreadMain_, reinterpret_cast<void*>(this));
#ifdef DEBUG
    DEBUG_ONLY(status_ = ATTACHED);
#endif
  }
  
  /**
   * Starts the thread running with specified priority.
   *
   * The priority number is backwards from priority -- higher numbers
   * have less priority.  Use a priority of 20 for the lowest possible
   * priority, or 0 if you don't want to change the priority.  Sorry, it is
   * not possible to increase your priority.
   */
  void Start(int prio) {
    pthread_attr_t tattr;
    sched_param param;

#ifdef DEBUG
    DEBUG_ASSERT(status_ == READY);
#endif
    pthread_attr_init(&tattr);
    pthread_attr_getschedparam(&tattr, &param);
    param.sched_priority = prio;
    pthread_attr_setschedparam(&tattr, &param);
    pthread_create(&thread_, &tattr,
        ThreadMain_, reinterpret_cast<void*>(this));
    pthread_attr_destroy(&tattr);
#ifdef DEBUG
    DEBUG_ONLY(status_ = ATTACHED);
#endif
  }
  
  /**
   * Detaches a thread -- the thread will cease to exist once the task
   * completes.  You may not call WaitStop on this thread afterwards.
   */
  void Detach() {
#ifdef DEBUG
    DEBUG_ASSERT(status_ == ATTACHED);
#endif
    pthread_detach(thread_);
#ifdef DEBUG
    DEBUG_ONLY(status_ = DETACHED);
#endif
  }
  
  /**
   * Wait for a thread to stop.
   *
   * Failure to do this may cause your program to hang when it is done.
   */
  void WaitStop() {
#ifdef DEBUG
    DEBUG_ASSERT(status_ == ATTACHED);
#endif
    pthread_join(thread_, NULL);
#ifdef DEBUG
    DEBUG_ONLY(status_ = DONE);
#endif
  }
  
  /**
   * Gets the contained task.
   */
  Task* task() const {
    return task_;
  }
};

/**
 * Mutual exclusion lock to protect shared data.
 */
class Mutex {
//  FORBID_ACCIDENTAL_COPIES(Mutex); // safe to remove?
  friend class WaitCondition;
 
 public:
  struct DummyRecursiveAttribute {};

 private:
  mutable pthread_mutex_t mutex_;

 public:
  static Mutex global;

 public:
  Mutex() {
#if defined(DEBUG) && defined(PTHREAD_ERRORCHECK_MUTEX_INITIALIZER_NP)
    mutex_ = (pthread_mutex_t)PTHREAD_ERRORCHECK_MUTEX_INITIALIZER_NP;
#else
    mutex_ = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
#endif
  }
  Mutex(DummyRecursiveAttribute v) {
#ifdef PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP
    mutex_ = (pthread_mutex_t)PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;
#else
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&mutex_, &attr);
    pthread_mutexattr_destroy(&attr);
#endif
  }

  ~Mutex() {
    pthread_mutex_destroy(&mutex_);
  }
  
  /** Obtains the lock. */
  void Lock() const {
    int t = pthread_mutex_lock(&mutex_);
    (void)t; // avoid an "unused variable" warning
    DEBUG_ASSERT_MSG(t == 0, "Error locking mutex -- relocking a non-recursive mutex?");
  }
  
  /** Tries to lock, returns false if doing so would require waiting. */
  bool TryLock() const {
    return likely(pthread_mutex_trylock(&mutex_) == 0);
  }
  
  /** Releases the lock. */
  void Unlock() const {
    pthread_mutex_unlock(&mutex_);
  }
};

/**
 * Mutual exclusion lock to protect shared data, but can be locked and
 * unlocked multiple times by the same thread without a deadlock.
 */
class RecursiveMutex : public Mutex {
  FORBID_ACCIDENTAL_COPIES(RecursiveMutex);

 public:
  RecursiveMutex() : Mutex(DummyRecursiveAttribute()) {}
};

/**
 * Wait condition for alerting other threads of an action.
 */
class WaitCondition {
  FORBID_ACCIDENTAL_COPIES(WaitCondition);
  
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
 * Reliable wait condition to signal readiness.
 *
 * This has semantics almost identical to the regular wait conditions,
 * except it will wake up only exactly one process, and that Wait() will
 * terminate immediately if Done() was previously called.
 *
 * This may be reused multiple times -- every time Wait() is called, the
 * done flag is reset to false afterwards.
 */
class DoneCondition {
  Mutex mutex_;
  WaitCondition cond_;
  bool done_;

 public:
  DoneCondition() { done_ = false; }
  ~DoneCondition() {}

  /**
   * Atomically waits for completion and then resets the done flag to false.
   */
  void Wait() {
    mutex_.Lock();
    while (!done_) {
      cond_.Wait(&mutex_);
    }
    done_ = false;
    mutex_.Unlock();
  }

  /**
   * Sets status to done and wakes up another process.
   */
  void Done() {
    mutex_.Lock();
    DEBUG_ASSERT_MSG(done_ == false, "Doesn't do a counter -- should it?");
    done_ = true;
    cond_.Signal();
    mutex_.Unlock();
  }
};

/**
 * Waits for a variable to take on a certain value.
 */
class ValueCondition {
  Mutex mutex_;
  WaitCondition cond_;
  int value_;

 public:
  ValueCondition() { value_ = 0; }
  ~ValueCondition() {}

  /**
   * Wait for this to become a particular value.
   */
  void Wait(int v) {
    mutex_.Lock();
    while (value_ != v) {
      cond_.Wait(&mutex_);
    }
    mutex_.Unlock();
  }

  void WaitNot(int v) {
    mutex_.Lock();
    while (value_ == v) {
      cond_.Wait(&mutex_);
    }
    mutex_.Unlock();
  }

  void Set(int v) {
    mutex_.Lock();
    if (value_ != v) {
      value_ = v;
      cond_.Broadcast();
    }
    mutex_.Unlock();
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
  FORBID_ACCIDENTAL_COPIES(Lockable);

  Lockable() {}
  ~Lockable() {}
};

#endif

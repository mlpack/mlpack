/**
 * Abstractions for arrays of pointerless structs.
 *
 * These arrays can be transparently distributed and parallel without
 * your slighted knowledge.
 */

#ifndef SUPERPAR_ARRAY_H
#define SUPERPAR_ARRAY_H

/**
 * Array abstraction for regular in-RAM arrays.
 *
 * This should not have any overhead over regular arrays.
 */
template<typename TElement>
class ArrayInCore {
  FORBID_COPY(ArrayInCore);

 public:
  /** The element type. */
  typedef TElement Element;

 private:
  /** A plain old in-memory array.
   *
   * TODO: Use ArrayList
   */
  Element *ptr_;
  /** The number of elements in the array. */
  index_t size_;
  /** Number of live items. */
  index_t live_;

 public:
#error "This doesn't cover region exclusion"
  ArrayInCore() {}
  ~ArrayInCore() {
    delete ptr_;
    DEBUG_ONLY(ptr_ = BIG_BAD_NUMBER);
    DEBUG_SAME_INT(live_, 0);
  }
  
  void Init(index_t size_in) {
    ptr_ = new Element[size_];
    size_ = size_in;
    live_ = 0;
  }
  
  index_t size() const {
    return size_;
  }
  
  /**
   * Efficient sampling from a large array.
   */
  void MakeSample(Element *dest_, index_t begin, index_t count,
      index_t n) {
    DEBUG_BOUNDS(n, size_);
    index_t step = count / n;
    index_t i_mine = begin;

    // TODO: Better sampling algorithm, or cache efficient algorithm
    for (index_t i_dest = 0; i_dest < n; i_dest++) {
      dest_[i_dest] = ptr_[i_mine];
      i_mine += step;
    }
  }

  /**
   * Starts reading from a segment of the array.
   *
   * This assumes that nobody else is writing to the same block
   * of memory, or that you are fine with unpredictable store/load
   * ordering.
   *
   * When done, you must call StopRead.
   */
  const Element *StartRead(index_t begin, index_t count) {
    DEBUG_BOUNDS(begin, size_);
    DEBUG_BOUNDS(begin + count, size_ + 1);
    DEBUG_ONLY(live_++);
    return ptr_ + begin;
  }

  /**
   * Stops reading from a segment of the array.
   *
   * This will free any resources associated with the returned
   * pointer (if necessary).
   */
  void StopRead(const Element *ptr, index_t begin, index_t count) {
    DEBUG_ASSERT(ptr - ptr_ == begin);
    DEBUG_ONLY(live_--);
    /* nothing necessary */
  }

  /**
   * Starts writing to a segment of the array.
   *
   * Exclusive access is required.  This returns a pointer that acts as
   * an array and can be written to.  When you are done, you must call
   * StopWrite.
   */
  Element *StartWrite(index_t begin, index_t count) {
    DEBUG_BOUNDS(begin, size_);
    DEBUG_BOUNDS(begin + count, size_ + 1);
    DEBUG_ONLY(live_++);
    return ptr_ + begin;
  }

  /**
   * Stops writing from a segment of the array.
   *
   * If necessary, this flushes back any changes and frees up any
   * associated resources.
   */
  void StopWrite(Element *ptr, index_t begin, index_t count) {
    DEBUG_ASSERT(ptr - ptr_ == begin);
    DEBUG_ONLY(live_--);
    /* nothing necessary */
  }

  /**
   * Starts reading a single element.
   *
   * When done, you must call StopRead.
   */
  const Element *StartRead(index_t element_id) {
    DEBUG_BOUNDS(element_id, size_);
    DEBUG_ONLY(live_++);
    return ptr_ + element_id;
  }

  /**
   * Stops reading a single element.
   *
   * This will free any resources associated with the returned
   * pointer (if necessary).
   */
  void StopRead(const Element *ptr, index_t element_id) {
    DEBUG_ASSERT(ptr - ptr_ == element_id);
    DEBUG_ONLY(live_--);
    /* nothing necessary */
  }

  /**
   * Starts writing to a segment of the array.
   *
   * Exclusive access is required.  This returns a pointer that acts as
   * an array and can be written to.  When you are done, you must call
   * StopWrite.
   */
  Element *StartWrite(index_t element_id) {
    DEBUG_BOUNDS(element_id, size_);
    DEBUG_ONLY(live_++);
    return ptr_ + element_id;
  }

  /**
   * Stops writing from a segment of the array.
   *
   * If necessary, this flushes back any changes and frees up any
   * associated resources.
   */
  void StopWrite(Element *ptr, index_t element_id) {
    DEBUG_ASSERT(ptr - ptr_ == element_id);
    DEBUG_ONLY(live_--);
    /* nothing necessary */
  }
};

#endif

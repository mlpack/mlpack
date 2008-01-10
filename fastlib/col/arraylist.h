// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file arraylist.h
 *
 * Typical bounds-checking resizable array implementation.
 */

#ifndef COLLECTIONS_ARRAYLIST_H
#define COLLECTIONS_ARRAYLIST_H

#include "base/base.h"

/**
 * Fast expandable array with debug-mode bounds checking.
 *
 * This has roughly similar features to std::vector.  However, an ArrayList
 * assumes that all objects can be relocated by just doing a shallow move
 * with realloc, without performing a deep copy on every element.
 * This means you cannot use an ArrayList if objects have
 * pointers to fields within themselves -- this isn't a very common
 * programming practice.  Like std::vector, it is unwise to have
 * external pointers to objects inside this array, if you expect that the
 * array might be resized.  On another note, this will initialize the memory
 * to "poison" values in debug mode, to make it easier to find problems
 * which otherwise would have undefined behavior.
 *
 * There are two typical usages: Knowing size ahead of time, and not knowing
 * the size.  If you don't know the size ahead of time:
 *
 * @code
 * // list of primitives
 * ArrayList<int> numbers;
 * numbers.Init();
 * while (some_condition) {
 *    *list.AddBack() = 42;
 * }
 * // list of objects
 * ArrayList<MyType> list;
 * list.Init();
 * while (some_condition) {
 *    list.AddBack()->Init(x, y, z);
 * }
 * @endcode
 *
 * If you know the size ahead of time:
 *
 * @code
 * ArrayList<MyType> list;
 * list.Init(55);
 * for (int i = 0; i < 55; i++) {
 *   list[i].Init(x, y, z);
 * }
 * @endcode
 *
 * In addition, ArrayList has all the definitions necessary for the object
 * traversal system, so it is suitable for use with THOR's automatic
 * serialization and deserialization.
 *
 */
template<typename TElement>
class ArrayList {
 public:
  /**
   * The element type.
   */
  typedef TElement Element;

 private:
  Element* ptr_;
  index_t size_;
  index_t cap_;

  OT_DEF_ONLY(ArrayList) {
    OT_MY_OBJECT(size_);
    OT_MALLOC_ARRAY(ptr_, size_);
  }

  OT_FIX(ArrayList) {
    cap_ = size_;
  }

 public:
  ArrayList() {
    DEBUG_ONLY(Invalidate_());
  }

  ArrayList(const ArrayList& other) {
    DEBUG_ONLY(Invalidate_());
    Copy(other);
  }
  ASSIGN_VIA_COPY_CONSTRUCTION(ArrayList);

  ~ArrayList() {
    Destruct();
  }

  /**
   * Returns this to an invalid state so it can be re-initialized.
   *
   * Example:
   *
   * @code
   * ArrayList<int> list;
   * list.Init(20);
   * ... do stuff with list
   * list.Destruct();
   * list.Copy(some_other_list);
   * @endcode
   */
  void Destruct() {
    DEBUG_ASSERT_MSG(ptr_ != BIG_BAD_POINTER(Element),
        "You forgot to initialize an ArrayList before it got automatically "
        "freed.  If you declare an ArrayList, you must use Init or similar, "
        "even if you never use it.");
    if (unlikely(ptr_ != NULL)) {
      mem::Destruct(ptr_, size_);
      mem::Free(ptr_);
    }
    DEBUG_ONLY(Invalidate_());
  }

  /**
   * Initialize an empty list.
   */
  void Init() {
    DEBUG_ASSERT_MSG(size_ == BIG_BAD_NUMBER, "reinitialization not allowed");
    ptr_ = NULL; /* yes, this will work */
    size_ = 0;
    cap_ = 0;
  }

  /**
   * Initializes to a given size.
   *
   * If you know the number of elements before hand, using this will save
   * both memory and CPU time.
   */
  void Init(index_t size_in) {
    Init(size_in, size_in);
  }

  /**
   * Initializes with a given size, but with a perhaps larger allocation.
   *
   * @param size_in the number of elements to start out with
   * @param cap_in what you expect the largest size it will grow to be
   */
  void Init(index_t size_in, index_t cap_in) {
    DEBUG_ASSERT_MSG(size_ == BIG_BAD_NUMBER, "reinitialization not allowed");
    DEBUG_ASSERT(size_in <= cap_in);

    size_ = size_in;
    cap_ = cap_in;

    ptr_ = mem::Alloc<Element>(cap_);
    // TODO: Default integer constructor initializes to zero; is there
    // a way to avoid this?
    mem::DefaultConstruct<Element>(ptr_, size_);
  }

  /**
   * Copies from another ArrayList.
   *
   * Requires the other list has a working copy constructor.
   */
  void Copy(const ArrayList& other) {
    Copy(other.ptr_, other.size_);
  }

  /**
   * Copies bit-for-bit from another array.
   *
   * Requires the other list has a working copy constructor and that your
   * data is validly bit-copiable.
   */
  void Copy(const Element *ptr, index_t size) {
    DEBUG_ASSERT_MSG(size_ == BIG_BAD_NUMBER, "reinitialization not allowed");

    ptr_ = mem::AllocCopyConstructed<Element>(ptr, size);
    cap_ = size;
    size_ = size;
  }

  /**
   * Resets to zero size and frees RAM.
   *
   * This is slower than Resize(0), since Resize(0) will hold onto the RAM
   * that was previously in use.
   */
  void Clear() {
    mem::Free(ptr_);
    ptr_ = NULL;
    size_ = 0;
    cap_ = 0;
  }

  /**
   * Steals the contents of another ArrayList, initializing this ArrayList and
   * making the other array list zero in size.
   */
  void Steal(ArrayList* other) {
    DEBUG_ASSERT_MSG(size_ == BIG_BAD_NUMBER, "reinitialization not allowed");

    ptr_ = other->ptr_;
    size_ = other->size_;
    cap_ = other->cap_;
    other->ptr_ = NULL;
    other->size_ = 0;
    other->cap_ = 0;
  }

  /**
   * Steals the contents of another ArrayList, initializing this ArrayList and
   * destructing the other ArrayList.
   *
   * WARNING: If the other ArrayList falls out of scope without being
   * reinitialized, the program will fail.
   */
  void StealDestruct(ArrayList* other) {
    DEBUG_ASSERT_MSG(size_ == BIG_BAD_NUMBER, "reinitialization not allowed");

    ptr_ = other->ptr_;
    size_ = other->size_;
    cap_ = other->cap_;

    DEBUG_ONLY(other->Invalidate_());
  }

  /**
   * Initializes this to a pointer allocated with mem::Alloc.
   *
   * It is assumed the first 'len' elements are constructed, and the rest
   * (up to 'capacity') are unconstructed.
   *
   * @param ptr a pointer allocated with mem::Alloc
   * @param len the number of constructed elementrs
   * @param capacity the total number of elements
   */
  void Steal(Element *ptr, index_t len, index_t capacity) {
    DEBUG_ASSERT_MSG(size_ == BIG_BAD_NUMBER, "reinitialization not allowed");

    ptr_ = ptr;
    size_ = len;
    cap_ = capacity;
  }

  /**
   * Returns the pointer to the beginning of the array, and reinitializes this
   * list to empty.
   */
  Element* ReleasePointer() {
    Element* retval = ptr_;
    ptr_ = NULL;
    size_ = 0;
    cap_ = 0;
    return retval;
  }

  /**
   * Switches the arrays pointed to by each array.
   */
  void Swap(ArrayList *other) {
    Element *t_ptr = other->ptr_;
    other->ptr_ = ptr_;
    ptr_ = t_ptr;

    index_t t_size = other->size_;
    other->size_ = size_;
    size_ = t_size;

    index_t t_cap = other->cap_;
    other->cap_ = cap_;
    cap_ = t_cap;
  }

  /**
   * Explicitly sets the size of the list.
   *
   * @param size_in the new size of the list
   */
  void Resize(index_t size_in) {
    if (likely(size_in > size_)) {
      IncreaseSizeHelper_(size_in);
    } else {
      DecreaseSizeHelper_(size_in);
    }
    size_ = size_in;
  }

  /**
   * Explicitly increases the size of the list.
   */
  void GrowTo(index_t size_in) {
    DEBUG_ASSERT(size_in >= size_);
    IncreaseSizeHelper_(size_in);
    size_ = size_in;
  }

  /**
   * Grows size if the current size isn't big enough.
   */
  void EnsureSizeAtLeast(index_t size_min) {
    if (unlikely(size_min > size_min)) {
      GrowTo(size_min);
    }
  }

  /**
   * Use this to shrink the ArrayList.
   */
  void DecreaseSize(index_t size_in) {
    DecreaseSizeHelper_(size_in);
    size_ = size_in;
  }

  /**
   * Use this to grow the ArrayList.
   */
  void IncreaseSize(index_t size_in) {
    IncreaseSizeHelper_(size_in);
    size_ = size_in;
  }

  /**
   * Use this to grow the size by a specified amount, returning a pointer
   * to the beginning of the chunk.
   */
  Element *AddBack(index_t size_increment) {
    if (unlikely(size_ + size_increment > cap_)) {
      IncreaseCap_(cap_ * 2 + size_increment);
    }

    Element* chunk = ptr_ + size_;

    size_ += size_increment;

    mem::DefaultConstruct(chunk, size_increment);

    return chunk;
  }

  /**
   * Adds one new element to the back, and returns the pointer to it.
   *
   * Example:
   *
   * @code
   * array_of_objects.AddBack()->Init(a, b, c);
   * *array_of_ints.AddBack() = 31;
   * @endcode
   *
   * WARNING: Don't make permanent pointers to this, as the array might
   * resize.
   *
   * TODO: Consider returning a mutable reference instead of a pointer, to
   * discourage people from making pointers to this object.
   *
   * @return a pointer to the newly created element
   */
  Element* AddBack() {
    if (unlikely(size_ == cap_)) {
      IncreaseCap_((cap_ + 1) * 2);
    }

    Element* elem = ptr_ + size_;

    ++size_;
    mem::DefaultConstruct(elem); // call default constructor

    return elem;
  }

  /**
   * Adds one specified element to the back, and returns the pointer to it.
   */
  Element* AddBackItem(const Element& value) {
    if (unlikely(size_ == cap_)) {
      IncreaseCap_((cap_ + 1) * 2);
    }

    Element* elem = ptr_ + size_;

    ++size_;
    new(elem)Element(value); // COPY CONSTRUCTOR! WOOT!

    return elem;
  }

  /**
   * Adds one specified element to the back,
   * BUT YOU MUST CALL ITS CONSTRUCTOR!
   */
  Element* AddBackUnconstructed() {
    if (unlikely(size_ == cap_)) {
      IncreaseCap_((cap_ + 1) * 2);
    }

    Element* elem = ptr_ + size_;

    ++size_;

    return elem;
  }

  /**
   * Removes the last element of the list.
   */
  void PopBack() {
    DEBUG_ASSERT(size_ > 0);
    --size_;
    mem::Destruct(ptr_ + size_);
  }

  /**
   * Returns a pointer to the last element, and decreases the size.
   *
   * Note that it is *your* responsibility to call the destructor (using
   * mem::Destruct or the destructor explicitly) of this
   * object if it is not a primitive.
   *
   * This will be invalidated if the ArrayList is subsequently trimmed.
   */
  Element* PopBackPtr() {
    --size_;

    return ptr_ + size_;
  }

  /**
   * Reallocates to the minimum memory usage to hold the data in the array.
   *
   * After lots of dynamic resizing, you may consider calling this.
   */
  void Trim() {
    DecreaseCap_(size_);
  }

  /**
   * Gets a constant element out of a constant ArrayList.
   */
  const Element& operator[] (index_t i) const {
    DEBUG_ASSERT_INDEX_BOUNDS(i, size_);
    return ptr_[i];
  }

  /**
   * Gets a mutable element out of an ArrayList.
   */
  Element& operator[] (index_t i) {
    DEBUG_ASSERT_INDEX_BOUNDS(i, size_);
    return ptr_[i];
  }

 public:
  /**
   * Gets the number of elements.
   */
  index_t size() const {
    return size_;
  }

  /**
   * Gets the number of elements this can hold without performing any
   * additional reallocations.
   */
  index_t capacity() const {
    return cap_;
  }

  /**
   * Returns a pointer to the first element.
   */
  const Element* begin() const {
    return ptr_;
  }

  /**
   * Returns a pointer one beyond the last element.
   */
  const Element* end() const {
    return ptr_ + size_;
  }

  /**
   * Returns a pointer to the last element.
   */
  const Element* last() const {
    return ptr_ + size_ - 1;
  }

  /**
   * Returns a pointer to the first element.
   */
  Element* begin() {
    return ptr_;
  }

  /**
   * Returns a pointer one beyond the last element.
   */
  Element* end() {
    return ptr_ + size_;
  }

  /**
   * Returns a pointer to the last element.
   */
  Element* last() {
    return ptr_ + size_ - 1;
  }

 private:
  /**
   * Increases the size by reallocating intelligently.
   */
  void IncreaseSizeHelper_(index_t size_in) {
    DEBUG_ASSERT(size_in >= size_);
    if (unlikely(size_in > cap_)) {
      IncreaseCap_(size_in + cap_);
    }
    mem::DefaultConstruct(ptr_ + size_, size_in - size_);
  }

  /**
   * Decreases the size, but doesn't actually resize.
   */
  void DecreaseSizeHelper_(index_t size_in) {
    DEBUG_ASSERT(size_in <= size_);
    mem::Destruct(ptr_ + size_in, size_ - size_in);
  }

  /**
   * Increases the array to be a larger size.
   *
   * This should not be inlined, because it is an unlikely case.
   */
  void IncreaseCap_(index_t cap_in);

  /**
   * Reallocates the array to be smaller.
   */
  void DecreaseCap_(index_t cap_in) {
    ptr_ = mem::Realloc(ptr_, cap_in);
    cap_ = cap_in;
  }

  /**
   * Sets fields to invalid values to ensure earliest possible catching of
   * debugging problems.
   */
  void Invalidate_() {
    DEBUG_POISON_PTR(ptr_);
    DEBUG_ONLY(size_ = BIG_BAD_NUMBER);
  }
};

template<typename TElement>
void ArrayList<TElement>::IncreaseCap_(index_t cap_in) {
  // round up capacities to sizeof(long)
  cap_in = (cap_in + sizeof(long) - 1) & ~(sizeof(long) - 1);
  ptr_ = mem::Realloc(ptr_, cap_in);
  cap_ = cap_in;
}

#endif

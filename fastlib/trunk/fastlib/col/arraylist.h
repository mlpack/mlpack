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
 * @file arraylist.h
 *
 * Defines a typical multi-purpose, resizable array and its utilities.
 */

#ifndef COL_ARRAYLIST_H
#define COL_ARRAYLIST_H

#include "../base/base.h"

#define ARRAYLIST__DEBUG_INIT_OK(who, size, cap) \
    DEBUG_INIT_OK(who); \
    DEBUG_ASSERT(size >= 0); \
    DEBUG_ASSERT(size <= cap);

#define ARRAYLIST__DEBUG_PUSH_BACK_OK(who, inc) \
    DEBUG_MODIFY_OK(who); \
    DEBUG_ASSERT(inc >= 0);

#define ARRAYLIST__DEBUG_POP_BACK_OK(who, dec) \
    DEBUG_MODIFY_OK(who); \
    DEBUG_BOUNDS_INCLUSIVE(dec, who->size());

#define ARRAYLIST__DEBUG_INSERT_OK(who, pos, inc) \
    DEBUG_MODIFY_OK(who); \
    DEBUG_BOUNDS_INCLUSIVE(pos, who->size()); \
    DEBUG_ASSERT(inc >= 0);

#define ARRAYLIST__DEBUG_REMOVE_OK(who, pos, dec) \
    DEBUG_MODIFY_OK(who); \
    DEBUG_BOUNDS_INCLUSIVE(pos, who->size()); \
    DEBUG_BOUNDS_INCLUSIVE(dec, who->size() - pos);

/**
 * A typical multi-purpose, resizable array, coded with an emphasis on
 * speed.  Also features debug-mode poisoning and bounds-checking.
 *
 * This class is roughly equivalent to std::vector, but with several
 * important differences.  Notably, it relocates objects with a simple
 * bit-copy, which is potentially much faster than std::vector's use
 * of the copy constructor but prevents use with objects that have
 * self-referential points (an uncommon programming practice).  Also,
 * ArrayList is fully FASTlib complient and can store all FASTlib
 * complient classes (i.e. those that declare OBJECT_TRAVERSAL) even
 * if they do not have copy constructors.
 *
 * As with std::vector, it is unwise to use pointers to objects stored
 * by ArrayList unless you know the array will not change for the
 * life-span of the pointer.
 */
template<typename TElem>
class ArrayList {
 public:
  /** An accessible typedef for the elements in the array. */
  typedef TElem Elem;

 private:
  Elem *ptr_;    // the stored or aliased array
  index_t size_; // number of active objects
  index_t cap_;  // allocated size of the array; -1 if alias

  OBJECT_TRAVERSAL_DEPRECATED_COPIES(ArrayList) {
    // note that only the active objects are copied, etc.
    OT_OBJ(size_);
    OT_ALLOC_EXPERT(ptr_, size_, false,
        i, OT_OBJ(ptr_[i]));
  }
  OT_DEFAULT_CONSTRUCT(ArrayList) {
    Reset_();
    OT__BecomeAlias_();
  }

  OT_TRANSIENTS(ArrayList) {
    OT_OBJ(cap_);
  }
  OT_REFILL_TRANSIENTS(ArrayList) {
    // after copy, array is truncated; also unsets aliasing
    cap_ = size_;
  }

  OT_BECOME_ALIAS(ArrayList) {
    cap_ = -1;
  }
  OT_IS_ALIAS(ArrayList) {
    return cap_ == -1;
  }
  OT_ALIAS_METHODS(ArrayList);

 private:
  /* Allocates more space; unlikely, so not inlined. */
  void IncreaseCap_(index_t cap);

  /* Brings the ArrayList to a default empty state. */
  void Reset_() {
    ptr_ = NULL;
    size_ = 0;
    cap_ = 0;
  }

 public:
  /**
   * Initializes an ArrayList with a given size and (optionally)
   * capacity (defaults to size) but does not construct the contents.
   *
   * This function can be used to initialize lists that will be
   * constructed later (e.g. by copy construction) and is needed for
   * the use of non-default constructors.  It is important to
   * construct or somehow fill the contents of the list before general
   * use or destruction.
   *
   * Note that shallow types (ints, doubles, etc. and types declared
   * with OBJECT_TRAVERSAL_SHALLOW) have NOP construction and thus no
   * performance gains come of using this function with them.
   *
   * @param size the initial number of elements
   * @param cap (optional) the initial amount of allocated space
   * @returns a pointer to the beginning of the array
   *
   * @see Init, InitCopy, InitRepeat
   */
  Elem *InitRaw(index_t size, index_t cap) {
    ARRAYLIST__DEBUG_INIT_OK(this, size, cap);
    ptr_ = mem::Alloc<Elem>(size);
    size_ = size;
    cap_ = cap;
    return ptr_;
  }
  Elem *InitRaw(index_t size) {
    return InitRaw(size, size);
  }

  /**
   * Initializes an empty ArrayList.
   *
   * @see InitCopy, InitRepeat, InitRaw, InitAlias, InitSteal
   */
  void Init() {
    DEBUG_INIT_OK(this);
    Reset_();
  }

  /**
   * Initializes an ArrayList with a given size and (optionally)
   * capacity (defaults to size), default constructing the contents.
   *
   * @param size the initial number of elements
   * @param cap (optional) the initial amount of allocated space
   *
   * @see InitCopy, InitRepeat, InitRaw
   */
  void Init(index_t size, index_t cap) {
    ARRAYLIST__DEBUG_INIT_OK(this, size, cap);
    ot::Construct(InitRaw(size, cap), size);
  }
  void Init(index_t size) {
    Init(size, size);
  }

  /**
   * Initializes an ArrayList with a given size and (optionally)
   * capacity (defaults to size), copying some object to each of the
   * elements.
   *
   * @param elem the initial value copied to each element
   * @param size the initial number of elements
   * @param cap (optional) the initial amount of allocated space
   *
   * @see Init, InitCopy, InitRaw
   */
  void InitRepeat(const Elem &elem, index_t size, index_t cap) {
    ARRAYLIST__DEBUG_INIT_OK(this, size, cap);
    ot::RepeatConstruct(InitRaw(size, cap), elem, size);
  }
  void InitRepeat(const Elem &elem, index_t size) {
    InitRepeat(elem, size, size);
  }

  /**
   * Initializes an ArrayList with a given size and (optionally)
   * capacity (defaults to size), element-wise copying the contents of
   * another array.  (Note that InitCopy of an ArrayList also exists.)
   *
   * @param ptr the initial values copied into the array
   * @param size the initial number of elements
   * @param cap (optional) the initial amount of allocated space
   *
   * @see Init, InitRepeat, InitRaw, InitAlias, InitSteal
   */
  void InitCopy(const Elem *src, index_t size, index_t cap) {
    ARRAYLIST__DEBUG_INIT_OK(this, size, cap);
    ot::CopyConstruct(InitRaw(size, cap), src, size);
  }
  void InitCopy(const Elem *src, index_t size) {
    InitCopy(src, size, size);
  }

  /**
   * Initializes an ArrayList to a copy of another ArrayList, but with
   * specified capacity.
   *
   * @param src the initial values copied into the array
   * @param cap the initial amount of allocated space
   *
   * @see Init, InitSubCopy
   */
  void InitCopy(const ArrayList &src, index_t cap) {
    ARRAYLIST__DEBUG_INIT_OK(this, src.size(), cap);
    InitCopy(src.begin(), src.size(), cap);
  }

  /**
   * Initializes an ArrayList to a copy of a section of another
   * ArrayList, with (optionally) specified capacity (defaults to the
   * other list's size).
   *
   * @param src the ArrayList to copy from
   * @param pos the offset into the source list
   * @param size the number of elements to copy
   * @param cap (optional) the initial amount of allocated space
   *
   * @see Init, InitCopy, InitSubAlias
   */
  void InitSubCopy(const ArrayList &src, index_t pos, index_t size,
		   index_t cap) {
    ARRAYLIST__DEBUG_INIT_OK(this, size, cap);
    DEBUG_BOUNDS_INCLUSIVE(pos, src.size());
    DEBUG_BOUNDS_INCLUSIVE(size, src.size() - pos);
    InitCopy(src.begin() + pos, size, capacity);
  }
  void InitSubCopy(const ArrayList &src, index_t pos, index_t size) {
    InitSubCopy(src, pos, size, size);
  }

  /**
   * Initializes an ArrayList as an alias of an array with a given
   * size.  (Note that InitAlias of an ArrayList also exists.)
   *
   * It is your responsibility to eventually free the aliased array,
   * but said array must not be freed or moved for the duration of the
   * alias's lifespan.
   *
   * @param ptr the array to alias
   * @param size the length of the aliased array
   *
   * @see Init, InitCopy, InitSubAlias
   */
  void InitAlias(Elem *ptr, index_t size) {
    DEBUG_INIT_OK(this);
    ptr_ = ptr;
    size_ = size;
    cap_ = -1;
  }

  /**
   * Initializes an ArrayList as an alias of a section of another
   * ArrayList.
   *
   * @param src the ArrayList to copy from
   * @param pos the offset into the source list
   * @param size the number of elements to copy
   *
   * @see Init, InitAlias, InitSubCopy
   */
  void InitSubAlias(const ArrayList &src, index_t pos, index_t size) {
    DEBUG_INIT_OK(this);
    DEBUG_BOUNDS_INCLUSIVE(pos, src.size());
    DEBUG_BOUNDS_INCLUSIVE(size, src.size() - pos);
    InitAlias(src.begin() + pos, size);
  }

  /**
   * Initializes an ArrayList directly with an array with a given size
   * and (optionally) capacity (defaults to size).  (Note that
   * InitSteal of an ArrayList also exists.)
   *
   * The input array will be automatically freed when the ArrayList is
   * destroyed and may be reallocated during use.  Because ArrayLists
   * use mem::Free and mem::Realloc, the array must be allocated with
   * mem::Alloc (or more appropriately, mem::AllocConstruct) rather
   * than new[].
   *
   * @param ptr the array to steal (assume deallocation rights)
   * @param size the number of elements in the stolen array
   * @param cap (optional) the allocated size of the stolen array
   *
   * @see Init, InitCopy, InitAlias
   */
  void InitSteal(Elem *ptr, index_t size, index_t cap) {
    ARRAYLIST__DEBUG_INIT_OK(this, size, cap);
    ptr_ = ptr;
    size_ = size;
    cap_ = cap;
  }
  void InitSteal(Elem *ptr, index_t size) {
    InitSteal(ptr, size, size);
  }



  /**
   * Returns a pointer to the beginning of the ArrayList's array and
   * demotes the ArrayList to an alias (voids deallocation rights).
   *
   * It is your responsibility to eventually free the returned array,
   * but said array must not be freed or moved for the duration of the
   * (now aliasing) ArrayList's lifespan.  Because ArrayLists use
   * mem::Alloc, the array must be deallocated with mem::Free.
   *
   * @see InitSteal
   */
  Elem *ReleasePtr() {
    DEBUG_MODIFY_OK(this);
    cap_ = -1;
    return ptr_;
  }

  /**
   * Exchanges arrays with another ArrayList.
   *
   * You can swap aliases and non-aliases safely.  Aliases made to
   * swapped arrays are not invalidated, but still point to the same
   * elements, i.e. which ArrayLists they alias are also conceptually
   * swapped.
   *
   * @param other the ArrayList to exchange with
   *
   * @see InitCopy, InitAlias, InitSteal
   */
  void Swap(ArrayList *other) {
    // note absense of DEBUG_MODIFY_OK => can swap aliases
    mem::Swap(this, other);
  }



  /**
   * Increases an ArrayList's size, increasing capacity if needed, and
   * default constructs newly added elements.
   *
   * This operation invalidates all pointers into the ArrayList
   * (including aliases) unless you can prove that the new size does
   * not exceed capacity, i.e. you Reserved space before creating the
   * pointers/aliases.
   *
   * @param size the increased size; must be larger than starting size
   *
   * @see ShrinkTo, Resize, Reserve
   */
  void GrowTo(index_t size) {
    DEBUG_MODIFY_OK(this);
    DEBUG_ASSERT(size >= size_);
    if (unlikely(size > cap_)) {
      IncreaseCap_(size + cap_);
    }
    ot::Construct(ptr_ + size_, size - size_);
    size_ = size;
  }

  /**
   * Decreases an ArrayList's size, destructing removed elements.
   *
   * This operation does not strictly invalidate pointers into the
   * ArrayList because it never reallocates to take less space, but
   * pointers to beyond the new size of the array are invalid because
   * they reference destructed objects.
   *
   * @param size the decreased size; must be less than starting size
   *
   * @see GrowTo, Resize, Reserve
   */
  void ShrinkTo(index_t size) {
    DEBUG_MODIFY_OK(this);
    DEBUG_BOUNDS_INCLUSIVE(size, size_);
    ot::Destruct(ptr_ + size, size_ - size);
    size_ = size;
  }

  /**
   * Changes an ArrayList's size, increasing capacity if needed, and
   * constructing or destructing elements as appropriate.
   *
   * This operation invalidates all pointers into the ArrayList
   * (including aliases) unless you can prove that the new size does
   * not exceed capacity, i.e. you Reserved space before creating the
   * pointers/aliases.
   *
   * @param size the new size for the array
   *
   * @see GrowTo, ShrinkTo, Reserve, Trim, Clear
   */
  void Resize(index_t size) {
    DEBUG_MODIFY_OK(this);
    DEBUG_ASSERT(size >= 0);
    if (unlikely(size > size_)) {
      GrowTo(size);
    } else {
      ShrinkTo(size);
    }
  }

  /**
   * Ensures that an ArrayList has at least size active elements,
   * adding and constructing elements as appropriate.
   *
   * This operation invalidates all pointers into the ArrayList
   * (including aliases) unless you can prove that the new size does
   * not exceed capacity, i.e. you Reserved space before creating the
   * pointers/aliases.
   *
   * @param size a lower bound on the size for the array
   *
   * @see Resize, GrowTo, Reserve
   */
  void SizeAtLeast(index_t size) {
    DEBUG_MODIFY_OK(this);
    DEBUG_ASSERT(size >= 0);
    if (unlikely(size > size_)) {
      GrowTo(size);
    }
  }

  /**
   * Ensures that an ArrayList has at most size active elements,
   * destructing and removing elements as appropriate.
   *
   * @param size an upper bound on the size for the array
   *
   * @see Resize, ShrinkTo, Trim
   */
  void SizeAtMost(index_t size) {
    DEBUG_MODIFY_OK(this);
    DEBUG_ASSERT(size >= 0);
    if (unlikely(size < size_)) {
      ShrinkTo(size);
    }
  }

  /**
   * Ensures that an ArrayList has capacity greater than a given size.
   *
   * Note that all operations that add elements to ArrayLists will
   * automatically increase capacity as needed.  This command's
   * purpose it to reduce the frequency of such operations in
   * hand-tuned situations or to ensure that pointers remain valid
   * after subsequent resizings (though this practice is error prone
   * and thus not recommended).
   *
   * This operation invalidates all pointers into the ArrayList
   * (including aliases).
   *
   * @param size the needed minimum capacity of the array
   *
   * @see Trim, Resize, Clear
   */
  void Reserve(index_t size) {
    DEBUG_MODIFY_OK(this);
    DEBUG_ASSERT(size >= 0);
    if (unlikely(size > cap_)) {
      ptr_ = mem::Realloc(ptr_, size);
      cap_ = size;
    }
  }

  /**
   * Reduces the capacity of an ArrayList to exactly its size.
   *
   * This may be useful to conserve memory in certain situations, but
   * keep in mind that copying, freezing, and serializing ArrayLists
   * already only uses as much space as is necessary to represent the
   * active objects.
   *
   * This operation invalidates all pointers into the ArrayList
   * (including aliases).
   *
   * @see Reserve, Resize, Clear
   */
  void Trim() {
    DEBUG_MODIFY_OK(this);
    ptr_ = mem::Realloc(ptr_, size_);
    cap_ = size_;
  }

  /**
   * Zeroes the size of an ArrayList and frees its memory.
   *
   * This brings an ArrayList to the same state as @c Init() and is
   * equivalent to @c Resize(0) followed by @c Trim() .  This is
   * slower than just @c Resize(0) because subsequent use needs to
   * allocate new memory.
   *
   * @see Resize, Trim, Reserve
   */
  void Clear() {
    DEBUG_MODIFY_OK(this);
    mem::Free(ot::Destruct(ptr_, size_));
    Reset_();
  }



  /**
   * Adds (optionally) inc elements (default 1) to the end of an
   * ArrayList, but does not construct them.
   *
   * It is important for you to construct or otherwise fill added
   * elements before you use them.
   *
   * Note that shallow types (ints, doubles, etc. and types declared
   * with OBJECT_TRAVERSAL_SHALLOW) have NOP construction and thus no
   * performance gains come of using this function with them.
   *
   * This operation invalidates all pointers into the ArrayList
   * (including aliases) unless you can prove that the new size does
   * not exceed capacity, i.e. you Reserved space before creating the
   * pointers/aliases.
   *
   * @param inc (optional) the number of elements to add
   * @returns a pointer to the first of the added elements
   *
   * @see PushBack, PushBackCopy, AppendCopy, PopBackRaw, InsertRaw
   */
  Elem *PushBackRaw(index_t inc = 1) {
    ARRAYLIST__DEBUG_PUSH_BACK_OK(this, inc);

    if (unlikely(size_ + inc > cap_)) {
      IncreaseCap_(size_ + inc + cap_);
    }

    Elem *elem = ptr_ + size_;
    size_ += inc;
    return elem;
  }

  /**
   * Adds (optionally) inc elements (default 1) to the end of an
   * ArrayList, default constructing them.
   *
   * Equivalent to @c a.GrowTo(a.size() + inc)
   *
   * This operation invalidates all pointers into the ArrayList
   * (including aliases) unless you can prove that the new size does
   * not exceed capacity, i.e. you Reserved space before creating the
   * pointers/aliases.
   *
   * @param inc (optional) the number of elements to add
   *
   * @see PushBackCopy, AppendCopy, PushBackRaw, PopBack, Insert,
   *      GrowTo
   */
  void PushBack(index_t inc) {
    ARRAYLIST__DEBUG_PUSH_BACK_OK(this, inc);
    ot::Construct(PushBackRaw(inc), inc);
  }
  Elem &PushBack() {
    DEBUG_MODIFY_OK(this);
    return *ot::Construct(PushBackRaw());
  }

  /**
   * Adds a copy of an element to the end of an ArrayList.
   *
   * This operation invalidates all pointers into the ArrayList
   * (including aliases) unless you can prove that the new size does
   * not exceed capacity, i.e. you Reserved space before creating the
   * pointers/aliases.
   *
   * @param src the element to add
   *
   * @see PushBack, AppendCopy, PopBackInit, InsertCopy
   */
  Elem &PushBackCopy(const Elem &src) {
    DEBUG_MODIFY_OK(this);
    return *ot::CopyConstruct(PushBackRaw(), &src);
  }

  /**
   * Adds copies of each element in an array to the end of an
   * ArrayList.
   *
   * This operation invalidates all pointers into the ArrayList
   * (including aliases) unless you can prove that the new size does
   * not exceed capacity, i.e. you Reserved space before creating the
   * pointers/aliases.
   *
   * @param src the array of elements to append
   * @param size the number of elements to append
   *
   * @see PushBackCopy, SegmentInit, InfixCopy
   */
  void AppendCopy(const Elem *src, index_t size) {
    ARRAYLIST__DEBUG_PUSH_BACK_OK(this, size);
    ot::CopyConstruct(PushBackRaw(size), src, size);
  }

  /**
   * Adds copies of each element in a source ArrayList to the end of
   * an ArrayList.
   *
   * This operation invalidates all pointers into the ArrayList
   * (including aliases) unless you can prove that the new size does
   * not exceed capacity, i.e. you Reserved space before creating the
   * pointers/aliases.
   *
   * @param src the Arraylist to append
   *
   * @see PushBackCopy, AppendSteal, SegmentInit, InfixCopy
   */
  void AppendCopy(const ArrayList &src) {
    DEBUG_MODIFY_OK(this);
    AppendCopy(src.begin(), src.size());
  }

  /**
   * Moves the elements in a source ArrayList to the end of an
   * ArrayList, converting the source into a sub-alias.
   *
   * This operation invalidates all pointers into the ArrayList
   * (including aliases) unless you can prove that the new size does
   * not exceed capacity, i.e. you Reserved space before creating the
   * pointers/aliases.
   *
   * Given the above, be careful not to AppendSteal from multiple
   * ArrayLists and then expect each of them to alias their respective
   * portions of the created ArrayList.
   *
   * @param src the ArrayList to append and steal from
   *
   * @see AppendCopy, InfixSteal
   */
  void AppendSteal(ArrayList *src);



  /**
   * Removes (optionally) dec elements (default 1) from the end of an
   * ArrayList, but does not destruct them.
   *
   * While it is technically possible to call this function and
   * destruct elements afterwards, it is recommended for you to
   * destruct elements first to minimize code change if you have to
   * move to RemoveRaw.
   *
   * Note that shallow types (ints, doubles, etc. and types declared
   * with OBJECT_TRAVERSAL_SHALLOW) have NOP destruction and thus no
   * performance gains come of using this function with them.
   *
   * @param dec (optional) the number of elements to remove
   *
   * @see PopBack, PopBackInit, SegmentInit, PushBackRaw, RemoveRaw
   */
  void PopBackRaw(index_t dec = 1) {
    ARRAYLIST__DEBUG_POP_BACK_OK(this, dec);
    size_ -= dec;
  }

  /**
   * Removes (optionally) dec elements (default 1) from the end of an
   * ArrayList, destructing them.
   *
   * Equivalent to @c a.ShrinkTo(a.size() - dec)
   *
   * @param dec (optional) the number of elements to remove
   *
   * @see PopBackInit, SegmentInit, PopBackRaw, PushBack, Remove,
   *      ShrinkTo
   */
  void PopBack(index_t dec = 1) {
    ARRAYLIST__DEBUG_POP_BACK_OK(this, dec);
    ot::Destruct(ptr_ + size_ - dec, dec);
    PopBackRaw(dec);
  }

  /**
   * Moves an element from the end of an ArrayList to a given location.
   *
   * Provided dest must have no clean-up responsibilities.  This is
   * always true of shallow types and generally true of travsered
   * types that have not been initialized, but can be complicated for
   * untraversed types that allocate on construction.  Consider
   * calling @c dest->~Elem() before this function, or just using the
   * element in place followed by a normal PopBack when finished.
   *
   * @param dest an uninitialized object to receive the element
   *
   * @see PopBack, SegmentInit, PushBackCopy, RemoveInit
   */
  void PopBackInit(Elem *dest) {
    DEBUG_MODIFY_OK(this);
    DEBUG_INIT_OK(dest);
    DEBUG_ASSERT(size_ > 0);
    mem::Copy(dest, ptr_ + size_ - 1);
    PopBackRaw();
  }

  /**
   * Moves multiple elements from the end of an ArrayList to a given
   * location.
   *
   * Provided dest must have no clean-up responsibilities.  This is
   * always true of shallow types and generally true of travsered
   * types that have not been initialized, but can be complicated for
   * untraversed types that allocate on construction.  Consider
   * calling @c dest[i].~Elem() before this function, or just using
   * the element in place followed by a normal PopBack when finished.
   *
   * @param size the number of elements to move
   * @param dest an array of uninitialized objects to receive the
   *        elements
   *
   * @see PopBackInit, AppendCopy, RemoveInit
   */
  void SegmentInit(index_t size, Elem *dest) {
    ARRAYLIST__DEBUG_POP_BACK_OK(this, size);
    mem::Copy(dest, ptr_ + size_ - size, size);
    PopBackRaw(size);
  }

  /**
   * Moves multiple elements from the end of an ArrayList to a new
   * ArrayList.
   *
   * @param size the number of elements to move
   * @param dest an uninitialized ArrayList to receive the elements
   *
   * @see PopBackInit, SegmentAppend, AppendCopy, ExtractInit
   */
  void SegmentInit(index_t size, ArrayList *dest);

  /**
   * Moves multiple elements from the end of an ArrayList to the end
   * of another ArrayList.
   *
   * @param size the number of elements to move
   * @param dest an ArrayList to receive the elements
   *
   * @see PopBackInit, SegmentInit, AppendCopy, ExtractAppend
   */
  void SegmentAppend(index_t size, ArrayList *dest);



  /**
   * Adds (optionally) inc elements (default 1) at a given position in
   * an ArrayList, but does not construct them.
   *
   * It is important for you to construct or otherwise fill added
   * elements before you use them.
   *
   * Note that shallow types (ints, doubles, etc. and types declared
   * with OBJECT_TRAVERSAL_SHALLOW) have NOP construction and thus no
   * performance gains come of using this function with them.
   *
   * This operation invalidates all pointers into the ArrayList
   * (including aliases) unless you can prove that the new size does
   * not exceed capacity, i.e. you Reserved space before creating the
   * pointers/aliases.
   *
   * @param pos the position where new elements should appear
   * @param inc (optional) the number of elements to add
   * @returns a pointer to the first of the added elements
   *
   * @see Insert, InsertCopy, InfixCopy, RemoveRaw, PushBackRaw
   */
  Elem *InsertRaw(index_t pos, index_t inc = 1) {
    ARRAYLIST__DEBUG_INSERT_OK(this, pos, inc);

    if (unlikely(size_ + inc > cap_)) {
      IncreaseCap_(size_ + inc + cap_);
    }

    mem::Move(ptr_ + pos + inc, ptr_ + pos, size_ - pos);
    size_ += inc;
    return ptr_ + pos;
  }

  /**
   * Adds (optionally) inc elements (default 1) at a given position in
   * an ArrayList, default constructing them.
   *
   * This operation invalidates all pointers into the ArrayList
   * (including aliases) unless you can prove that the new size does
   * not exceed capacity, i.e. you Reserved space before creating the
   * pointers/aliases.
   *
   * @param pos the position where new elements should appear
   * @param inc (optional) the number of elements to add
   *
   * @see InsertCopy, InfixCopy, InsertRaw, Remove, PushBack
   */
  void Insert(index_t pos, index_t inc) {
    ARRAYLIST__DEBUG_INSERT_OK(this, pos, inc);
    ot::Construct(InsertRaw(pos, inc), inc);
  }
  Elem &Insert(index_t pos) {
    DEBUG_MODIFY_OK(this);
    DEBUG_BOUNDS_INCLUSIVE(pos, size_);
    return *ot::Construct(InsertRaw(pos));
  }

  /**
   * Adds a copy of an element at a given position in an ArrayList.
   *
   * This operation invalidates all pointers into the ArrayList
   * (including aliases) unless you can prove that the new size does
   * not exceed capacity, i.e. you Reserved space before creating the
   * pointers/aliases.
   *
   * @param pos the position where the new element should appear
   * @param src the element to add
   *
   * @see Insert, InfixCopy, RemoveInit, PushBackCopy
   */
  Elem &InsertCopy(index_t pos, const Elem &src) {
    DEBUG_MODIFY_OK(this);
    DEBUG_BOUNDS_INCLUSIVE(pos, size_);
    return *ot::CopyConstruct(InsertRaw(pos), &src);
  }

  /**
   * Adds copies of each element in an array at a given position in an
   * ArrayList.
   *
   * This operation invalidates all pointers into the ArrayList
   * (including aliases) unless you can prove that the new size does
   * not exceed capacity, i.e. you Reserved space before creating the
   * pointers/aliases.
   *
   * @param pos the position where infixed elements should appear
   * @param src the array of elements to infix
   * @param size the number of elements to infix
   *
   * @see InsertCopy, ExtractInit, AppendCopy
   */
  void InfixCopy(index_t pos, const Elem *src, index_t size) {
    ARRAYLIST__DEBUG_INSERT_OK(this, pos, size);
    ot::CopyConstruct(InsertRaw(pos, size), src, size);
  }

  /**
   * Adds copies of each element in a source ArrayList at a given
   * position in an ArrayList.
   *
   * This operation invalidates all pointers into the ArrayList
   * (including aliases) unless you can prove that the new size does
   * not exceed capacity, i.e. you Reserved space before creating the
   * pointers/aliases.
   *
   * @param pos the position where infixed elements should appear
   * @param src the Arraylist to append
   *
   * @see InsertCopy, InfixSteal, ExtractInit, AppendCopy
   */
  void InfixCopy(index_t pos, const ArrayList &src) {
    DEBUG_MODIFY_OK(this);
    DEBUG_BOUNDS_INCLUSIVE(pos, size_);
    InfixCopy(pos, src.begin(), src.size());
  }

  /**
   * Moves the elements in a source ArrayList to a given position in
   * an ArrayList, converting the source into a sub-alias.
   *
   * This operation invalidates all pointers into the ArrayList
   * (including aliases) unless you can prove that the new size does
   * not exceed capacity, i.e. you Reserved space before creating the
   * pointers/aliases.
   *
   * Given the above, be careful not to InfixSteal from multiple
   * ArrayLists and then expect each of them to alias their respective
   * portions of the created ArrayList, especially because infixed
   * regions may ultimately (and inappropriately) overlap.
   *
   * @param pos the position where infixed elements should appear
   * @param src the ArrayList to infix and steal from
   *
   * @see AppendCopy, InfixSteal
   */
  void InfixSteal(index_t pos, ArrayList *src);



  /**
   * Removes (optionally) dec elements (default 1) from a given
   * position in an ArrayList, but does not destruct them.
   *
   * It is important for you to destruct elements before calling this
   * function, as they will be overwritten.
   *
   * Note that shallow types (ints, doubles, etc. and types declared
   * with OBJECT_TRAVERSAL_SHALLOW) have NOP destruction and thus no
   * performance gains come of using this function with them.
   *
   * @param pos the position of the first element to remove
   * @param dec (optional) the number of elements to remove
   *
   * @see Remove, RemoveInit, ExtractInit, InsertRaw, PopBackRaw
   */
  void RemoveRaw(index_t pos, index_t dec = 1) {
    ARRAYLIST__DEBUG_REMOVE_OK(this, pos, dec);
    mem::Move(ptr_ + pos, ptr_ + pos + dec, size_ - dec - pos);
    size_ -= dec;
  }

  /**
   * Removes (optionally) dec elements (default 1) from a given
   * position in an ArrayList, destructing them.
   *
   * @param pos the position of the first element to remove
   * @param dec (optional) the number of elements to remove
   *
   * @see RemoveInit, ExtractInit, RemoveRaw, Insert, PopBack
   */
  void Remove(index_t pos, index_t dec = 1) {
    ARRAYLIST__DEBUG_REMOVE_OK(this, pos, dec);
    ot::Destruct(ptr_ + pos, dec);
    RemoveRaw(pos, dec);
  }

  /**
   * Moves an element from a given position in an ArrayList to a given
   * location.
   *
   * Provided dest must have no clean-up responsibilities.  This is
   * always true of shallow types and generally true of travsered
   * types that have not been initialized, but can be complicated for
   * untraversed types that allocate on construction.  Consider
   * calling @c dest->~Elem() before this function, or just using the
   * element in place followed by a normal Remove when finished.
   *
   * @param pos the position of the element to move
   * @param dest an uninitialized object to receive the element
   *
   * @see Remove, ExtractInit, InsertCopy, PopBackInit
   */
  void RemoveInit(index_t pos, Elem *dest) {
    DEBUG_MODIFY_OK(this);
    DEBUG_INIT_OK(dest);
    DEBUG_BOUNDS(pos, size_);
    mem::Copy(dest, ptr_ + pos);
    RemoveRaw(pos);
  }

  /**
   * Moves multiple elements from a given position in an ArrayList to
   * a given location.
   *
   * Provided dest must have no clean-up responsibilities.  This is
   * always true of shallow types and generally true of travsered
   * types that have not been initialized, but can be complicated for
   * untraversed types that allocate on construction.  Consider
   * calling @c dest[i].~Elem() before this function, or just using
   * the element in place followed by a normal Remove when finished.
   *
   * @param pos the position of the first element to move
   * @param size the number of elements to move
   * @param dest an array of uninitialized objects to receive the
   *        elements
   *
   * @see RemoveInit, InfixCopy, PopBackInit
   */
  void ExtractInit(index_t pos, index_t size, Elem *dest) {
    ARRAYLIST__DEBUG_REMOVE_OK(this, pos, size);
    mem::Copy(dest, ptr_ + pos, size);
    RemoveRaw(pos, size);
  }

  /**
   * Moves multiple elements from a given position in an ArrayList to
   * a new ArrayList.
   *
   * @param pos the position of the first element to move
   * @param size the number of elements to move
   * @param dest an uninitialized ArrayList to receive the elements
   *
   * @see RemoveInit, ExtractAppend, InfixCopy, SegmentInit
   */
  void ExtractInit(index_t pos, index_t size, ArrayList *dest);

  /**
   * Moves multiple elements from a given position in an ArrayList to
   * the end of another ArrayList.
   *
   * @param pos the position of the first element to move
   * @param size the number of elements to move
   * @param dest an ArrayList to receive the elements
   *
   * @see RemoveInit, ExtractInit, InfixCopy, SegmentAppend
   */
  void ExtractAppend(index_t pos, index_t size, ArrayList *dest);



  /** The number of active elements in the ArrayList. */
  index_t size() const {
    return size_;
  }
  /** The number of active elements in the ArrayList. 
   * This function is defined so that it provides the same 
   * interface for lapack/blas operations*/
  index_t length() const {
    return this->size();
  }

  /** The allocated number of elements, or -1 if alias. */
  index_t capacity() const {
    return cap_;
  }
  /** Whether the ArrayList is empty (size 0). */
  bool empty() const {
    return size_ == 0;
  }

  /** Access an element at position i. */
  const Elem &operator[] (index_t i) const {
    DEBUG_BOUNDS(i, size_);
    return ptr_[i];
  }
  Elem &operator[] (index_t i) {
    DEBUG_BOUNDS(i, size_);
    return ptr_[i];
  }

  /** Get a pointer to the beginning of the array. */
  const Elem *begin() const {
    return ptr_;
  }
  Elem* begin() {
    return ptr_;
  }

  /** Get a pointer to just past the end of the array. */
  const Elem *end() const {
    return ptr_ + size_;
  }
  Elem *end() {
    return ptr_ + size_;
  }

  /** Get the first element in the array, as in @c *a.begin() */
  const Elem &front() const {
    return *ptr_;
  }
  Elem &front() {
    return *ptr_;
  }

  /** Get the last element in the array, as in @c *(a.end() - 1) */
  const Elem &back() const {
    return ptr_[size_ - 1];
  }
  Elem &back() {
    return ptr_[size_ - 1];
  }

  ////////// Deprecated //////////////////////////////////////////////

  COMPILER_DEPRECATED_MSG("Renamed InitCopy")
  void Copy(const Elem *src, index_t size) {
    InitCopy(src, size);
  }
  COMPILER_DEPRECATED_MSG("Renamed InitSteal")
  void Steal(const Elem *src, index_t size) {
    InitSteal(src, size);
  }
  COMPILER_DEPRECATED_MSG("Renamed InitSteal; other will alias")
  void Steal(ArrayList *other) {
    InitSteal(other);
    other->Reset_();
  }

  COMPILER_DEPRECATED_MSG("Renamed ReleasePtr; will become alias")
  Elem *ReleasePointer() {
    Elem *retval = ReleasePtr();
    Reset_();
    return retval;
  }
  COMPILER_DEPRECATED_MSG("Renamed Renew")
  void Destruct() {
    Renew();
  }

  COMPILER_DEPRECATED_MSG("Renamed SizeAtLeast")
  void EnsureSizeAtLeast(index_t size) {
    SizeAtLeast(size);
  }

  COMPILER_DEPRECATED_MSG("Renamed PushBack; no longer returns pointer")
  Elem *AddBack(index_t inc = 1) {
    index_t offset = size_;
    PushBack(inc);
    return ptr_ + offset;
  }
  COMPILER_DEPRECATED_MSG("Renamed PushBackRaw")
  Elem *AddBackUnconstructed(index_t inc = 1) {
    return PushBackRaw(inc);
  }
  COMPILER_DEPRECATED_MSG("Renamed PushBackCopy")
  Elem *AddBackItem(const Elem &elem) {
    return &PushBackCopy(elem);
  }

  COMPILER_DEPRECATED_MSG("Use PopBackInit instead")
  Elem *PopBackPtr() {
    PopBackRaw();
    return end();
  }
};

template<typename TElem>
void ArrayList<TElem>::IncreaseCap_(index_t cap) {
  // round up capcity for possible paging performance
  cap = (cap + sizeof(long) - 1) & ~(sizeof(long) - 1);
  ptr_ = mem::Realloc(ptr_, cap);
  cap_ = cap;
}

template<typename TElem>
void ArrayList<TElem>::AppendSteal(ArrayList *src) {
  DEBUG_MODIFY_OK(this);
  DEBUG_MODIFY_OK(src);

  Elem *elem = PushBackRaw(src->size());
  mem::Copy(elem, src->begin(), src->size());

  mem::Free(src->ptr_);
  src->ptr_ = elem;
  src->cap_ = -1;
}

template<typename TElem>
void ArrayList<TElem>::SegmentInit(index_t size, ArrayList *dest) {
  ARRAYLIST__DEBUG_POP_BACK_OK(this, size);
  DEBUG_INIT_OK(dest);
  SegmentInit(size, dest->InitRaw(size));
}

template<typename TElem>
void ArrayList<TElem>::SegmentAppend(index_t size, ArrayList *dest) {
  ARRAYLIST__DEBUG_POP_BACK_OK(this, size);
  DEBUG_MODIFY_OK(dest);
  SegmentInit(size, dest->PushBackRaw(size));
}

template<typename TElem>
void ArrayList<TElem>::InfixSteal(index_t pos, ArrayList *src) {
  DEBUG_MODIFY_OK(this);
  DEBUG_MODIFY_OK(src);
  DEBUG_BOUNDS_INCLUSIVE(pos, size_);

  Elem *elem = InsertRaw(pos, src->size());
  mem::Copy(elem, src->begin(), src->size());

  mem::Free(src->ptr_);
  src->ptr_ = elem;
  src->cap_ = -1;
}

template<typename TElem>
void ArrayList<TElem>::ExtractInit(index_t pos, index_t size,
				   ArrayList *dest) {
  ARRAYLIST__DEBUG_REMOVE_OK(this, pos, size);
  DEBUG_INIT_OK(dest);
  ExtractInit(pos, size, dest->InitRaw(size));
}

template<typename TElem>
void ArrayList<TElem>::ExtractAppend(index_t pos, index_t size,
				     ArrayList *dest) {
  ARRAYLIST__DEBUG_REMOVE_OK(this, pos, size);
  DEBUG_MODIFY_OK(dest);
  ExtractInit(pos, size, dest->PushBackRaw(size));
}

#undef ARRAYLIST__DEBUG_REMOVE_OK
#undef ARRAYLIST__DEBUG_INSERT_OK
#undef ARRAYLIST__DEBUG_POP_BACK_OK
#undef ARRAYLIST__DEBUG_PUSH_BACK_OK
#undef ARRAYLIST__DEBUG_INIT_OK

#endif /* COL_ARRAYLIST_H */

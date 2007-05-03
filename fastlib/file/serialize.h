// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file serialize.h
 *
 * Tools for serialization.
 *
 * See NativeArraySerializer and NativeArrayDeserializer for an example of
 * the Serializer and Deserializer "concepts".
 *
 * WARNING: This has been only casually tested, not thoroughly tested.
 * Use at your own risk.
 */

#ifndef FILE_SERIALIZE_H
#define FILE_SERIALIZE_H

#include "base/common.h"
#include "base/ccmem.h"
#include "col/arraylist.h"

#include <typeinfo>

namespace file {
  /**
   * Magic number type.
   */
  typedef uint32 magic_t;
  
  /**
   * Automatically creates a magic number by hashing the
   * string.
   */
  magic_t CreateMagic(const char *str);
};

/**
 * Automatically creates a magic number for a class.
 *
 * This is especially useful if you have a templated class.
 * You can add other magic numbers to it if you want the number to be
 * more specific, like MAGIC_NUMBER(a) + MAGIC_NUMBER(b).
 *
 * @param classname the name of the class
 */
#define MAGIC_NUMBER(classname) \
    (file::CreateMagic(typeid(classname).name()))

/**
 * Use this to serialize a stream of objects in an array.
 *
 * Useful before sending canonical versions over MPI, or to a file.
 *
 * The computer reading this should use the same CPU type.
 * In particular, the type sizes and byte order must remain the same.
 *
 * Only primitives or structs of primitives should be stored; in
 * particular, no pointers.
 *
 * TODO: More complicated serializers (like text serializers) later
 * might have a slight problem with serializing structs.  Come back
 * later for info.  Maybe we'll decide against text serializers.
 */
class NativeArraySerializer {
  FORBID_COPY(NativeArraySerializer);
  
 private:
  ArrayList<char> data_;
  
 public:
  NativeArraySerializer() {}
  ~NativeArraySerializer() {}

  void Init() {
    data_.Init();
  }
 
  void PutMagic(file::magic_t magic_num) {
    Put(magic_num);
  }

  /**
   * Appends a struct or primitive to the end.
   *
   * This will store the object bit-by-bit, so don't store any
   * pointers.
   */
  template<typename T>
  void Put(const T& val) {
    mem::CopyBytes(data_.AddBack(sizeof(T)), &val, sizeof(T));
  }

  /**
   * Appends an array of structs or primtives to the end.
   *
   * This will store the object bit-by-bit, so don't store any
   * pointers.
   */
  template<typename T>
  void Put(const T* array, index_t count) {
    size_t bytes = count * sizeof(T);
    mem::CopyBytes(data_.AddBack(bytes), array, bytes);
  }

  /**
   * Appends an array of non-primitives to the end.
   */
  template<typename T>
  void Serialize(const T* array, index_t count) {
    for (index_t i = 0; i < count; i++) {
      array[i].Serialize(this);
    }
  }
  
  /**
   * Returns the data serialized, clearing the internal
   * state of this serializer.
   *
   * The pointer must eventually be freed with mem::Free.
   *
   * @return a pointer that can later be passed to deserializer
   */
  char *ReleasePointer() {
    data_.Trim();
    return data_.ReleasePointer();
  }
  
  const char *ptr() const {
    return data_.begin();
  }
  
  index_t size() const {
    return data_.size();
  }
  
  /**
   * Dumps to a file.
   */
  success_t WriteFile(const char *fname);
};

/**
 * Use this to deserialize a stream created by NativeArraySerializer.
 *
 * This computer should use the same architecture as was used to seralize.
 * Specifically, the type sizes and byte order must be identical.
 */
class NativeArrayDeserializer {  
 private:
  const char *ptr_;
  index_t pos_;
  index_t size_;
  
 public:
  NativeArrayDeserializer() {
    DEBUG_POISON_PTR(ptr_);
  }
  ~NativeArrayDeserializer() {}
  
  /**
   * Initializes given a pointer.
   *
   * The pointer will not be freed.
   */
  void Init(const char *ptr_in, index_t size_in) {
    ptr_ = ptr_in;
    size_ = size_in;
    pos_ = 0;
  }
  
  success_t CheckMagic(file::magic_t magic_num) {
    file::magic_t x;
    Get(&x);
    return likely(x == magic_num) ? SUCCESS_PASS : SUCCESS_FAIL;
  }
  
  void AssertMagic(file::magic_t magic_num) {
    file::magic_t x;
    Get(&x);
    assert(x == magic_num);
  }
  
  /**
   * Retrieves the next struct or primtiive.
   */
  template<typename T>
  void Get(T *dest) {
    DEBUG_ASSERT(size_t(pos_ + sizeof(T)) <= size_t(size_));
    mem::CopyBytes(dest, ptr_ + pos_, sizeof(T));
    pos_ += sizeof(T);
  }
  
  /**
   * Retrieves the next array of structs or primitives.
   */
  template<typename T>
  void Get(T *dest, index_t count) {
    size_t bytes = count * sizeof(T);
    DEBUG_ASSERT(index_t(pos_ + bytes) <= index_t(size_));
    mem::CopyBytes(dest, ptr_ + pos_, bytes);
    pos_ += bytes;
  }
  
  /**
   * Retrieves an array of Serializables.
   */
  template<typename T>
  void Deserialize(T *dest, index_t count) {
    for (index_t i = 0; i < count; i++) {
      dest[i].Deserialize(this);
    }
  }
  
  /**
   * Returns whether finished.
   *
   * You are not recommended to use this; store counts instead.
   *
   * @return true iff there is no data remaining
   */
  bool done() const {
    return pos_ == size_;
  }
  
  size_t size() const {
    return size_;
  }
  
  size_t pos() const {
    return pos_;
  }
  
  const char *ptr() const {
    return ptr_;
  }
};

/**
 * Deserializer from a file.
 *
 * NOTE: This currently uses a NativeArrayDeserializer, but in the future
 * may use buffering.
 */
class NativeFileDeserializer : public NativeArrayDeserializer {
 public:
  NativeFileDeserializer() {}
  ~NativeFileDeserializer() {
    mem::Free(const_cast<char*>(ptr()));
  }
  
  /**
   * Initializes by reading the entire file into memory.
   */
  success_t Init(const char *fname);
};

#endif

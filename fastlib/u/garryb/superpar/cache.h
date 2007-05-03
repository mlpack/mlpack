


template<typename TElement>
class CacheArray {
  FORBID_COPY(CacheArray);

 public:
  typedef TElement Element;

 private:
  Element *ptr_;
  index_t size_;
  index_t live_;

 public:
  CacheArray() {}
  
  ~CacheArray() {
    delete ptr_;
    DEBUG_ONLY(ptr_ = BIG_BAD_NUMBER);
    DEBUG_SAME_INT(live_, 0);
  }
  
  void Init(index_t size_in) {
  }
  
  index_t size() const {
  }

  const Element *StartRead(index_t element_id) {
  }

  void StopRead(const Element *ptr, index_t element_id) {
  }

  Element *StartWrite(index_t element_id) {
  }

  void StopWrite(Element *ptr, index_t element_id) {
  }
  
  void DeclareWritebackRange(index_t start, index_t count) {
  }
  
  void DeclareTempRange(index_t start, index_t count) {
  }
  
  void FlushWritebackRange(index_t start, index_t count) {
  }
};

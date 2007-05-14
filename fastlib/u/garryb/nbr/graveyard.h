



/*

if (main) {
  q_points = new CacheArray(new SmallCache(new BlankBackend()));
  register(q_points);
  ... fill in data ...
  q_tree = new CacheArray(new SmallCache(new BlankBackend()));
  register(q_tree);
  ... fill in tree ...
  my_q_results = new CacheArray(new SmallCache(new BlankBackend()));
  (this really is the only machine that knows its default values)
  initialize the elements of my_q_results
  register(my_q_results)
}

wait_for_main

q_points = new CacheArray(new SmallCache(new NetBackend("q_points")));
q_tree = new SmallCache(new NetBackend("q_tree")));
q_results = new SmallCache(new NetBackend("q_results")));
q_temps = new CacheArray();

spawn threads

if (main) {
  write my_q_results to file
}

here's a cache
here's your backend
*/

/*

template<typename Freezer>
class CacheArray {
  FORBID_COPY(CacheArray);

 public:
  typedef TElement Element;

 private:
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
*/

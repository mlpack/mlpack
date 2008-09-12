(** Arrays. Extension of ExtLib's ExtArray, which itself extends Standard Library's Array. *)

exception Different_array_size
  (** Raised by functions taking two or more arrays that should have same length but are given arrays of different length. Analagous to ExtLib's Different_list_size. *)
  
external length : 'a array -> int = "%array_length"
    (** Return the length (number of elements) of the given array. *)
    
external get : 'a array -> int -> 'a = "%array_safe_get"
    (** [Array.get a n] returns the element number [n] of array [a].
        The first element has number 0.
        The last element has number [Array.length a - 1].
        You can also write [a.(n)] instead of [Array.get a n].
        
        Raise [Invalid_argument "index out of bounds"]
        if [n] is outside the range 0 to [(Array.length a - 1)]. *)
    
external set : 'a array -> int -> 'a -> unit = "%array_safe_set"
    (** [Array.set a n x] modifies array [a] in place, replacing
        element number [n] with [x].
        You can also write [a.(n) <- x] instead of [Array.set a n x].
        
        Raise [Invalid_argument "index out of bounds"]
        if [n] is outside the range 0 to [Array.length a - 1]. *)
    
    
(** {6 Constructors} *)
    
external make : int -> 'a -> 'a array = "caml_make_vect"
    (** [Array.make n x] returns a fresh array of length [n],
        initialized with [x].
        All the elements of this new array are initially
        physically equal to [x] (in the sense of the [==] predicate).
        Consequently, if [x] is mutable, it is shared among all elements
        of the array, and modifying [x] through one of the array entries
        will modify all other entries at the same time.
        
        Raise [Invalid_argument] if [n < 0] or [n > Sys.max_array_length].
        If the value of [x] is a floating-point number, then the maximum
        size is only [Sys.max_array_length / 2].*)
    
val init : int -> (int -> 'a) -> 'a array
  (** [Array.init n f] returns a fresh array of length [n],
      with element number [i] initialized to the result of [f i].
      In other terms, [Array.init n f] tabulates the results of [f]
      applied to the integers [0] to [n-1].
      
      Raise [Invalid_argument] if [n < 0] or [n > Sys.max_array_length].
      If the return type of [f] is [float], then the maximum
      size is only [Sys.max_array_length / 2].*)
  
val make_matrix : int -> int -> 'a -> 'a array array
  (** [Array.make_matrix dimx dimy e] returns a two-dimensional array
      (an array of arrays) with first dimension [dimx] and
      second dimension [dimy]. All the elements of this new matrix
      are initially physically equal to [e].
      The element ([x,y]) of a matrix [m] is accessed
      with the notation [m.(x).(y)].
      
      Raise [Invalid_argument] if [dimx] or [dimy] is negative or
      greater than [Sys.max_array_length].
      If the value of [e] is a floating-point number, then the maximum
      size is only [Sys.max_array_length / 2]. *)
  
val append : 'a array -> 'a array -> 'a array
  (** [Array.append v1 v2] returns a fresh array containing the
      concatenation of the arrays [v1] and [v2]. *)
  
val concat : 'a array list -> 'a array
  (** Same as [Array.append], but concatenates a list of arrays. *)
  
val sub : 'a array -> int -> int -> 'a array
  (** [Array.sub a start len] returns a fresh array of length [len],
      containing the elements number [start] to [start + len - 1]
      of array [a].
      
      Raise [Invalid_argument "Array.sub"] if [start] and [len] do not
      designate a valid subarray of [a]; that is, if
      [start < 0], or [len < 0], or [start + len > Array.length a]. *)
  
val copy : 'a array -> 'a array
  (** [Array.copy a] returns a copy of [a], that is, a fresh array
      containing the same elements as [a]. *)
  
val fill : 'a array -> int -> int -> 'a -> unit
  (** [Array.fill a ofs len x] modifies the array [a] in place,
      storing [x] in elements number [ofs] to [ofs + len - 1].
      
      Raise [Invalid_argument "Array.fill"] if [ofs] and [len] do not
      designate a valid subarray of [a]. *)
  
val blit : 'a array -> int -> 'a array -> int -> int -> unit
  (** [Array.blit v1 o1 v2 o2 len] copies [len] elements
      from array [v1], starting at element number [o1], to array [v2],
      starting at element number [o2]. It works correctly even if
      [v1] and [v2] are the same array, and the source and
      destination chunks overlap.
      
      Raise [Invalid_argument "Array.blit"] if [o1] and [len] do not
      designate a valid subarray of [v1], or if [o2] and [len] do not
      designate a valid subarray of [v2]. *)

val of_idx_array : (int * 'a) array -> 'a -> 'a array
  (** [of_idx_array idxa default] treats [idxa] as an association of indices with values. Return a fresh array where the [i]th value is set to [v] if the input has a pair [(i,v)]. Size of returned array will be maximum index in input + 1. Indices not given a value in [idxa] will be set to [default] value. If there are duplicate indices in [idxa], the last value will override former ones. *)
  
  
(** {6 Converters} *)
  
val rev : 'a array -> 'a array
  (** Array reversal. *)
  
val rev_in_place : 'a array -> unit
  (** In-place array reversal.  The array argument is updated. *)
  
val to_list : 'a array -> 'a list
  (** Convert an array to a list. *)
  
val of_list : 'a list -> 'a array
  (** Convert a list to an array. *)
  
val of_list2 : 'a list list -> 'a array array
val to_list2 : 'a array array -> 'a list list
  
val zip : 'a array -> 'b array -> ('a * 'b) array
  (** [zip a b] pairs up values in [a] and [b]. Order of elements is preserved. Raise {!Different_array_size} if [a] and [b] do not have the same length. *)
  
val unzip : ('a * 'b) array -> ('a array * 'b array)
  (** [unzip ab] returns two arrays [a] and [b], where [a] has all the first elements of the pairs in [ab], and [b] has the second elements. Order of elements is preserved. *)
  
val enum : 'a array -> 'a Enum.t
  (** Returns an enumeration of the elements of an array. *)
  
val of_enum : 'a Enum.t -> 'a array
  (** Build an array from an enumeration. *)
  
  
(** {6 Iterators} *)
  
val iter : ('a -> unit) -> 'a array -> unit
  (** [Array.iter f a] applies function [f] in turn to all
      the elements of [a].  It is equivalent to
      [f a.(0); f a.(1); ...; f a.(Array.length a - 1); ()]. *)
  
val map : ('a -> 'b) -> 'a array -> 'b array
  (** [Array.map f a] applies function [f] to all the elements of [a],
      and builds an array with the results returned by [f]:
      [[| f a.(0); f a.(1); ...; f a.(Array.length a - 1) |]]. *)
  
val iteri : (int -> 'a -> unit) -> 'a array -> unit
  (** Same as {!Array.iter}, but the
      function is applied to the index of the element as first argument,
      and the element itself as second argument. *)
  
val mapi : (int -> 'a -> 'b) -> 'a array -> 'b array
  (** Same as {!Array.map}, but the
      function is applied to the index of the element as first argument,
      and the element itself as second argument. *)
  
val fold_left : ('a -> 'b -> 'a) -> 'a -> 'b array -> 'a
  (** [Array.fold_left f x a] computes
      [f (... (f (f x a.(0)) a.(1)) ...) a.(n-1)],
      where [n] is the length of the array [a]. *)
  
val fold_right : ('b -> 'a -> 'a) -> 'b array -> 'a -> 'a
  (** [Array.fold_right f a x] computes
      [f a.(0) (f a.(1) ( ... (f a.(n-1) x) ...))],
      where [n] is the length of the array [a]. *)
  

(** {6 Scanning} *)
  
val mem : 'a -> 'a array -> bool
  (** [mem m a] is true if and only if [m] is equal to an element of [a]. *)
  
val memq : 'a -> 'a array -> bool
  (** Same as {!Array.mem} but uses physical equality instead of
      structural equality to compare array elements.
  *)
  
val for_all : ('a -> bool) -> 'a array -> bool
  (** [for_all p [a1; ...; an]] checks if all elements of the array
      satisfy the predicate [p].  That is, it returns
      [ (p a1) && (p a2) && ... && (p an)].
  *)
  
val for_alli : (int -> 'a -> bool) -> 'a array -> bool
  
val exists : ('a -> bool) -> 'a array -> bool
  (** [exists p [a1; ...; an]] checks if at least one element of
      the array satisfies the predicate [p].  That is, it returns
      [ (p a1) || (p a2) || ... || (p an)].
  *)
  
val existsi : (int -> 'a -> bool) -> 'a array -> bool
  
val find : ('a -> bool) -> 'a array -> 'a
  (** [find p a] returns the first element of array [a]
      that satisfies the predicate [p].
      Raise [Not_found] if there is no value that satisfies [p] in the
      array [a].
  *)
  
val unique : ('a -> 'a -> int) -> 'a array -> bool
  (** [unique comp t] returns true if no two items in the array are equal as determined by [comp]. *)
  
val is_rectangular : 'a array array -> bool
  (** [is_rectangular a] returns true if length of every a.(i) is the same. *)
  
  
(** {6 Searching} *)
  
val findi : ('a -> bool) -> 'a array -> int
  (** [findi p a] returns the index of the first element of array [a]
      that satisfies the predicate [p].
      Raise [Not_found] if there is no value that satisfies [p] in the
      array [a].
  *)
  
val find' : (int -> 'a -> bool) -> 'a array -> 'a
val findi' : (int -> 'a -> bool) -> 'a array -> int
  (** similar to ExtLib's [find] and [findi], but the predicate can also employ the index of the item. *) 
  
val filter : ('a -> bool) -> 'a array -> 'a array
  (** [filter p a] returns all the elements of the array [a]
      that satisfy the predicate [p].  The order of the elements
      in the input array is preserved.  *)
  
val find_all : ('a -> bool) -> 'a array -> 'a array
  (** [find_all] is another name for {!Array.filter}. *)
  
val partition : ('a -> bool) -> 'a array -> 'a array * 'a array
  (** [partition p a] returns a pair of arrays [(a1, a2)], where
      [a1] is the array of all the elements of [a] that
      satisfy the predicate [p], and [a2] is the array of all the
      elements of [a] that do not satisfy [p].
      The order of the elements in the input array is preserved. *)
  
(** {6 Sorting} *)    
  
val sort : ('a -> 'a -> int) -> 'a array -> unit
  (** Sort an array in increasing order according to a comparison
      function.  The comparison function must return 0 if its arguments
      compare as equal, a positive integer if the first is greater,
      and a negative integer if the first is smaller (see below for a
      complete specification).  For example, {!Pervasives.compare} is
      a suitable comparison function, provided there are no floating-point
      NaN values in the data.  After calling [Array.sort], the
      array is sorted in place in increasing order.
      [Array.sort] is guaranteed to run in constant heap space
      and (at most) logarithmic stack space.
      
      The current implementation uses Heap Sort.  It runs in constant
      stack space.
      
      Specification of the comparison function:
      Let [a] be the array and [cmp] the comparison function.  The following
      must be true for all x, y, z in a :
      -   [cmp x y] > 0 if and only if [cmp y x] < 0
      -   if [cmp x y] >= 0 and [cmp y z] >= 0 then [cmp x z] >= 0
      
      When [Array.sort] returns, [a] contains the same elements as before,
      reordered in such a way that for all i and j valid indices of [a] :
      -   [cmp a.(i) a.(j)] >= 0 if and only if i >= j
  *)
  
val stable_sort : ('a -> 'a -> int) -> 'a array -> unit
  (** Same as {!Array.sort}, but the sorting algorithm is stable (i.e.
      elements that compare equal are kept in their original order) and
      not guaranteed to run in constant heap space.
      
      The current implementation uses Merge Sort. It uses [n/2]
      words of heap space, where [n] is the length of the array.
      It is usually faster than the current implementation of {!Array.sort}.
  *)
  
val fast_sort : ('a -> 'a -> int) -> 'a array -> unit
  (** Same as {!Array.sort} or {!Array.stable_sort}, whichever is faster
      on typical input.
  *)
  
  
(**/**)
(** {6 Undocumented functions} *)
  
external unsafe_get : 'a array -> int -> 'a = "%array_unsafe_get"
external unsafe_set : 'a array -> int -> 'a -> unit = "%array_unsafe_set"

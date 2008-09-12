(** Lists. Extension of ExtLib's ExtList, which itself extends Standard Library's List. *)

val length : 'a list -> int
  (** Return the length (number of elements) of the given list. *)
  
val hd : 'a list -> 'a
  (** Return the first element of the given list. Raise
      [Failure "hd"] if the list is empty. *)
  
val tl : 'a list -> 'a list
  (** Return the given list without its first element. Raise
      [Failure "tl"] if the list is empty. *)
  
val nth : 'a list -> int -> 'a
  (** Return the [n]-th element of the given list.
      The first element (head of the list) is at position 0.
      Raise [Failure "nth"] if the list is too short.
      Raise [Invalid_argument "List.nth"] if [n] is negative. *)

val first : 'a list -> 'a
  (** Returns the first element of the list, or raise [Empty_list] if
      the list is empty (similar to [hd]). *)
  
val last : 'a list -> 'a
  (** Returns the last element of the list, or raise [Empty_list] if
      the list is empty. This function takes linear time. *)
  

(** {6 Constructors} *)
  
val init : int -> (int -> 'a) -> 'a list
  (** Similar to [Array.init], [init n f] returns the list containing
      the results of (f 0),(f 1).... (f (n-1)).
      Raise [Invalid_arg "ExtList.init"] if n < 0.*)
  
val make : int -> 'a -> 'a list
  (** Similar to [String.make], [make n x] returns a
      * list containing [n] elements [x].
  *)
  
val append : 'a list -> 'a list -> 'a list
  (** Catenate two lists.  Same function as the infix operator [@].
      Not tail-recursive (length of the first argument).  The [@]
      operator is not tail-recursive either. *)
  
val rev_append : 'a list -> 'a list -> 'a list
  (** [List.rev_append l1 l2] reverses [l1] and concatenates it to [l2].
      This is equivalent to {!List.rev}[ l1 @ l2], but [rev_append] is
      tail-recursive and more efficient. *)
  
val concat : 'a list list -> 'a list
  (** Concatenate a list of lists.  The elements of the argument are all
      concatenated together (in the same order) to give the result.
      Not tail-recursive
      (length of the argument + length of the longest sub-list). *)
  
val flatten : 'a list list -> 'a list
  (** Same as [concat].  Not tail-recursive
      (length of the argument + length of the longest sub-list). *)
  
val merge : ('a -> 'a -> int) -> 'a list -> 'a list -> 'a list
  (** Merge two lists:
      Assuming that [l1] and [l2] are sorted according to the
      comparison function [cmp], [merge cmp l1 l2] will return a
      sorted list containting all the elements of [l1] and [l2].
      If several elements compare equal, the elements of [l1] will be
      before the elements of [l2].
      Not tail-recursive (sum of the lengths of the arguments).
  *)

  
(** {6 Converters} *)

val zip : 'a list -> 'b list -> ('a * 'b) list
val zip3 : 'a list -> 'b list -> 'c list -> ('a * 'b * 'c) list
val zip4 : 'a list -> 'b list -> 'c list -> 'd list -> ('a * 'b * 'c * 'd) list

val unzip : ('a * 'b) list -> ('a list * 'b list)
val unzip3 : ('a * 'b * 'c) list -> ('a list * 'b list * 'c list)
val unzip4 : ('a * 'b * 'c * 'd) list -> ('a list * 'b list * 'c list * 'd list)

val rev : 'a list -> 'a list
  (** List reversal. *)
  
val split : ('a * 'b) list -> 'a list * 'b list
  (** Transform a list of pairs into a pair of lists:
      [split [(a1,b1); ...; (an,bn)]] is [([a1; ...; an], [b1; ...; bn])].
      Not tail-recursive.
  *)
  
val combine : 'a list -> 'b list -> ('a * 'b) list
  (** Transform a pair of lists into a list of pairs:
      [combine [a1; ...; an] [b1; ...; bn]] is
      [[(a1,b1); ...; (an,bn)]].
      Raise [Invalid_argument] if the two lists
      have different lengths.  Not tail-recursive. *)
  
val take : int -> 'a list -> 'a list
  (** [take n l] returns up to the [n] first elements from list [l], if
      available. *)
  
val drop : int -> 'a list -> 'a list
  (** [drop n l] returns [l] without the first [n] elements, or the empty
      list if [l] have less than [n] elements. *)
  
val takewhile : ('a -> bool) -> 'a list -> 'a list
  (** [takewhile f xs] returns the first elements of list [xs]
      which satisfy the predicate [f]. *)
  
val dropwhile : ('a -> bool) -> 'a list -> 'a list
  (** [dropwhile f xs] returns the list [xs] with the first
      elements satisfying the predicate [f] dropped. *)
  
val interleave : 'a list -> 'a list -> 'a list
val to_string : ('a -> string) -> 'a list -> string
  
val enum : 'a list -> 'a Enum.t
  (** Returns an enumeration of the elements of a list. *)
  
val of_enum : 'a Enum.t -> 'a list
  (** Build a list from an enumeration. *)
  
  
(** {6 Iterators} *)
  
val iter : ('a -> unit) -> 'a list -> unit
  (** [List.iter f [a1; ...; an]] applies function [f] in turn to
      [a1; ...; an]. It is equivalent to
      [begin f a1; f a2; ...; f an; () end]. *)
  
val map : ('a -> 'b) -> 'a list -> 'b list
  (** [List.map f [a1; ...; an]] applies function [f] to [a1, ..., an],
      and builds the list [[f a1; ...; f an]]
      with the results returned by [f].  Not tail-recursive. *)
  
val iteri : (int -> 'a -> 'b) -> 'a list -> unit
  (** [iteri f l] will call [(f 0 a0);(f 1 a1) ... (f n an)] where
      [a0..an] are the elements of the list [l]. *)
  
val mapi : (int -> 'a -> 'b) -> 'a list -> 'b list
  (** [mapi f l] will build the list containing
      [(f 0 a0);(f 1 a1) ... (f n an)] where [a0..an] are the elements of
      the list [l]. *)
  
val rev_map : ('a -> 'b) -> 'a list -> 'b list
  (** [List.rev_map f l] gives the same result as
      {!List.rev}[ (]{!List.map}[ f l)], but is tail-recursive and
      more efficient. *)
  
val fold_left : ('a -> 'b -> 'a) -> 'a -> 'b list -> 'a
  (** [List.fold_left f a [b1; ...; bn]] is
      [f (... (f (f a b1) b2) ...) bn]. *)
  
val fold_right : ('a -> 'b -> 'b) -> 'a list -> 'b -> 'b
  (** [List.fold_right f [a1; ...; an] b] is
      [f a1 (f a2 (... (f an b) ...))].  Not tail-recursive. *)
  
val iter2 : ('a -> 'b -> unit) -> 'a list -> 'b list -> unit
  (** [List.iter2 f [a1; ...; an] [b1; ...; bn]] calls in turn
      [f a1 b1; ...; f an bn].
      Raise [Invalid_argument] if the two lists have
      different lengths. *)
  
val map2 : ('a -> 'b -> 'c) -> 'a list -> 'b list -> 'c list
  (** [List.map2 f [a1; ...; an] [b1; ...; bn]] is
      [[f a1 b1; ...; f an bn]].
      Raise [Invalid_argument] if the two lists have
      different lengths.  Not tail-recursive. *)
  
val rev_map2 : ('a -> 'b -> 'c) -> 'a list -> 'b list -> 'c list
  (** [List.rev_map2 f l1 l2] gives the same result as
      {!List.rev}[ (]{!List.map2}[ f l1 l2)], but is tail-recursive and
      more efficient. *)
  
val fold_left2 : ('a -> 'b -> 'c -> 'a) -> 'a -> 'b list -> 'c list -> 'a
  (** [List.fold_left2 f a [b1; ...; bn] [c1; ...; cn]] is
      [f (... (f (f a b1 c1) b2 c2) ...) bn cn].
      Raise [Invalid_argument] if the two lists have
      different lengths. *)
  
val fold_right2 : ('a -> 'b -> 'c -> 'c) -> 'a list -> 'b list -> 'c -> 'c
  (** [List.fold_right2 f [a1; ...; an] [b1; ...; bn] c] is
      [f a1 b1 (f a2 b2 (... (f an bn c) ...))].
      Raise [Invalid_argument] if the two lists have
      different lengths.  Not tail-recursive. *)
  

(** {6 Scanning} *)
  
val for_all : ('a -> bool) -> 'a list -> bool
  (** [for_all p [a1; ...; an]] checks if all elements of the list
      satisfy the predicate [p]. That is, it returns
      [(p a1) && (p a2) && ... && (p an)]. *)
  
val exists : ('a -> bool) -> 'a list -> bool
  (** [exists p [a1; ...; an]] checks if at least one element of
      the list satisfies the predicate [p]. That is, it returns
      [(p a1) || (p a2) || ... || (p an)]. *)
  
val for_all2 : ('a -> 'b -> bool) -> 'a list -> 'b list -> bool
  (** Same as {!List.for_all}, but for a two-argument predicate.
      Raise [Invalid_argument] if the two lists have
      different lengths. *)
  
val exists2 : ('a -> 'b -> bool) -> 'a list -> 'b list -> bool
  (** Same as {!List.exists}, but for a two-argument predicate.
      Raise [Invalid_argument] if the two lists have
      different lengths. *)
  
val mem : 'a -> 'a list -> bool
  (** [mem a l] is true if and only if [a] is equal
      to an element of [l]. *)
  
val memq : 'a -> 'a list -> bool
  (** Same as {!List.mem}, but uses physical equality instead of structural
      equality to compare list elements. *)
  
val elements_unique : ?cmp:('a -> 'a -> bool) -> 'a list -> bool
  (** Return true if elements in list are unique. Default [cmp] is =. *)
  
(** {6 Searching} *)
  
val find : ('a -> bool) -> 'a list -> 'a
  (** [find p l] returns the first element of the list [l]
      that satisfies the predicate [p].
      Raise [Not_found] if there is no value that satisfies [p] in the
      list [l]. *)
  
val filter : ('a -> bool) -> 'a list -> 'a list
  (** [filter p l] returns all the elements of the list [l]
      that satisfy the predicate [p].  The order of the elements
      in the input list is preserved.  *)
  
val find_all : ('a -> bool) -> 'a list -> 'a list
  (** [find_all] is another name for {!List.filter}. *)
  
val rfind : ('a -> bool) -> 'a list -> 'a
  (** [rfind p l] returns the last element [x] of [l] such as [p x] returns
      [true] or raises [Not_found] if such element as not been found. *)
  
val find_exc : ('a -> bool) -> exn -> 'a list -> 'a
  (** [find_exc p e l] returns the first element of [l] such as [p x]
      returns [true] or raises [e] if such element as not been found. *)
  
val findi : (int -> 'a -> bool) -> 'a list -> (int * 'a)
  (** [findi p e l] returns the first element [ai] of [l] along with its
      index [i] such that [p i ai] is true, or raises [Not_found] if no
      such element has been found. *)
  
val unique : ?cmp:('a -> 'a -> bool) -> 'a list -> 'a list
  (** [unique cmp l] returns the list [l] without any duplicate element.
      Default comparator ( = ) is used if no comparison function specified. *)
  
val filter_map : ('a -> 'b option) -> 'a list -> 'b list
  (** [filter_map f l] call [(f a0) (f a1).... (f an)] where [a0..an] are
      the elements of [l]. It returns the list of elements [bi] such as
      [f ai = Some bi] (when [f] returns [None], the corresponding element of
      [l] is discarded). *)
  
val split_nth : int -> 'a list -> 'a list * 'a list
  (** [split_nth n l] returns two lists [l1] and [l2], [l1] containing the
      first [n] elements of [l] and [l2] the others. Raise [Invalid_index] if
      [n] is outside of [l] size bounds. *)
  
val remove : 'a list -> 'a -> 'a list
  (** [remove l x] returns the list [l] without the first element [x] found
      or returns  [l] if no element is equal to [x]. Elements are compared
      using ( = ). *)
  
val remove_if : ('a -> bool) -> 'a list -> 'a list
  (** [remove_if cmp l] is similar to [remove], but with [cmp] used
      instead of ( = ). *)
  
val remove_all : 'a list -> 'a -> 'a list
  (** [remove_all l x] is similar to [remove] but removes all elements that
      are equal to [x] and not only the first one. *)
  
val partition : ('a -> bool) -> 'a list -> 'a list * 'a list
  (** [partition p l] returns a pair of lists [(l1, l2)], where
      [l1] is the list of all the elements of [l] that
      satisfy the predicate [p], and [l2] is the list of all the
      elements of [l] that do not satisfy [p].
      The order of the elements in the input list is preserved. *)

val npartition : ('a -> 'a -> bool) -> 'a list -> 'a list list
  (** [npartition eq l] paritions input list [l] into lists [\[l1; l2; ...; ln\]], such that elements within each [li] are equal according to [eq], and any two elements from two different lists are not equal. Within each returned list, order of elements in original list is preserved. *)

val first_repeat : ?cmp:('a -> 'a -> bool) -> 'a list -> 'a
  (** Return first repeated item in given list, or raise [Not_found] if all elements unique. Default [cmp] is [(=)]. *)  
  
(** {6 Association lists} *)
  
val assoc : 'a -> ('a * 'b) list -> 'b
  (** [assoc a l] returns the value associated with key [a] in the list of
      pairs [l]. That is,
      [assoc a [ ...; (a,b); ...] = b]
      if [(a,b)] is the leftmost binding of [a] in list [l].
      Raise [Not_found] if there is no value associated with [a] in the
      list [l]. *)
  
val assq : 'a -> ('a * 'b) list -> 'b
  (** Same as {!List.assoc}, but uses physical equality instead of structural
      equality to compare keys. *)
  
val mem_assoc : 'a -> ('a * 'b) list -> bool
  (** Same as {!List.assoc}, but simply return true if a binding exists,
      and false if no bindings exist for the given key. *)
  
val mem_assq : 'a -> ('a * 'b) list -> bool
  (** Same as {!List.mem_assoc}, but uses physical equality instead of
      structural equality to compare keys. *)
  
val remove_assoc : 'a -> ('a * 'b) list -> ('a * 'b) list
  (** [remove_assoc a l] returns the list of
      pairs [l] without the first pair with key [a], if any.
      Not tail-recursive. *)
  
val remove_assq : 'a -> ('a * 'b) list -> ('a * 'b) list
  (** Same as {!List.remove_assoc}, but uses physical equality instead
      of structural equality to compare keys.  Not tail-recursive. *)

val set_assoc_with : ('b option -> 'b) -> 'a -> ('a * 'b) list -> ('a * 'b) list
  (** [set_assoc_with f a l] searches [l] for the first item with key [a]. If found, it replaces the value [b] with [f (Some b)]. If there is no item with key [a], the new association [(a, f None)] is inserted at the end. Key comparison uses [=]. *)

  
(** {6 Sorting} *)

val is_sorted : ?cmp:('a -> 'a -> int) -> 'a list -> bool
  (** Return true if list is sorted according to [cmp] (default is {!Pervasives.compare}). *)
  
val sort : ?cmp:('a -> 'a -> int) -> 'a list -> 'a list
  (** Sort a list in increasing order according to a comparison
      function.  The comparison function must return 0 if its arguments
      compare as equal, a positive integer if the first is greater,
      and a negative integer if the first is smaller (see Array.sort for
      a complete specification).  For example,
      {!Pervasives.compare} is a suitable comparison function.
      The resulting list is sorted in increasing order.
      [List.sort] is guaranteed to run in constant heap space
      (in addition to the size of the result list) and logarithmic
      stack space.
      
      The current implementation uses Merge Sort. It runs in constant
      heap space and logarithmic stack space.
  *)
  
val stable_sort : ('a -> 'a -> int) -> 'a list -> 'a list
  (** Same as {!List.sort}, but the sorting algorithm is guaranteed to
      be stable (i.e. elements that compare equal are kept in their
      original order) .
      
      The current implementation uses Merge Sort. It runs in constant
      heap space and logarithmic stack space.
  *)
  
val fast_sort : ('a -> 'a -> int) -> 'a list -> 'a list
  (** Same as {!List.sort} or {!List.stable_sort}, whichever is faster
      on typical input. *)


(** {6 Exceptions} *)

exception Empty_list
  (** [Empty_list] is raised when an operation applied on an empty list
      is invalid : [hd] for example. *)
  
exception Invalid_index of int
  (** [Invalid_index] is raised when an indexed access on a list is
      out of list bounds. *)
  
exception Different_list_size of string
  (** [Different_list_size] is raised when applying functions such as
      [iter2] on two lists having different size. *)

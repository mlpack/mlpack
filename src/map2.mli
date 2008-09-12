(** Association tables over ordered types.

    This module implements applicative association tables, also known as
    finite maps or dictionaries, given a total ordering function
    over the keys.
    All operations over maps are purely applicative (no side-effects).
    The implementation uses balanced binary trees, and therefore searching
    and insertion take time logarithmic in the size of the map.
    
    Extension of Standard Library's Map.
*)

module type OrderedType =
sig
  type t
      (** The type of the map keys. *)
      
  val compare : t -> t -> int
    (** A total ordering function over the keys.
        This is a two-argument function [f] such that
        [f e1 e2] is zero if the keys [e1] and [e2] are equal,
        [f e1 e2] is strictly negative if [e1] is smaller than [e2],
        and [f e1 e2] is strictly positive if [e1] is greater than [e2].
        Example: a suitable ordering function is the generic structural
        comparison function {!Pervasives.compare}. *)
end
  (** Input signature of the functor {!Map.Make}. *)
  
module type S =
sig
  type key
      (** The type of the map keys. *)
      
  type (+'a) t
      (** The type of maps from type [key] to type ['a]. *)

  val is_empty: 'a t -> bool
    (** Test whether a map is empty or not. *)
    
  val size : 'a t -> int
    (** Retrun number of bindings in the map. *)

  val compare: ('a -> 'a -> int) -> 'a t -> 'a t -> int
    (** Total ordering between maps.  The first argument is a total ordering
        used to compare data associated with equal keys in the two maps. *)
    
  val equal: ('a -> 'a -> bool) -> 'a t -> 'a t -> bool
    (** [equal cmp m1 m2] tests whether the maps [m1] and [m2] are
        equal, that is, contain equal keys and associate them with
        equal data.  [cmp] is the equality predicate used to compare
        the data associated with the keys. *)


  (** {6 Constructors and Modifiers} *)
  
  val empty: 'a t
    (** The empty map. *)
    
  val add: key -> 'a -> 'a t -> 'a t
    (** [add x y m] returns a map containing the same bindings as
        [m], plus a binding of [x] to [y]. If [x] was already bound
        in [m], its previous binding disappears. *)
    
  val remove: key -> 'a t -> 'a t
    (** [remove x m] returns a map containing the same bindings as
        [m], except for [x] which is unbound in the returned map. *)
    

  (** {6 Convertors} *)
    
  val of_array : (key * 'a) array -> 'a t
  val of_list : (key * 'a) list -> 'a t
    (** Construct map from array/list of (key,value) pairs. If there are duplicate keys in input, the last item is the one inserted. *)

  val to_array : 'a t -> (key * 'a) array
  val to_list : 'a t -> (key * 'a) list
    (** Returned array/list has the (key,value) pairs in given map. Items will be in ascending order by key. *)

    
  (** {6 Iterators} *)
    
  val iter: (key -> 'a -> unit) -> 'a t -> unit
    (** [iter f m] applies [f] to all bindings in map [m].
        [f] receives the key as first argument, and the associated value
        as second argument.  The bindings are passed to [f] in increasing
        order with respect to the ordering over the type of the keys.
        Only current bindings are presented to [f]:
        bindings hidden by more recent bindings are not passed to [f]. *)
    
  val map: ('a -> 'b) -> 'a t -> 'b t
    (** [map f m] returns a map with same domain as [m], where the
        associated value [a] of all bindings of [m] has been
        replaced by the result of the application of [f] to [a].
        The bindings are passed to [f] in increasing order
        with respect to the ordering over the type of the keys. *)

  val map2 : ('a -> 'b -> 'c) -> 'a t -> 'b t -> 'c t
    (** [map2 f m n] is like [map] but operates on two maps. Raise [Failure] if the domains of maps [m] and [n] are not equal. *)
    
  val mapi: (key -> 'a -> 'b) -> 'a t -> 'b t
    (** Same as {!Map.S.map}, but the function receives as arguments both the
        key and the associated value for each binding of the map. *)
    
  val map2i : (key -> 'a -> 'b -> 'c) -> 'a t -> 'b t -> 'c t
    (** Like [map2] but the function also receives the key as an argument. *)
    
  val fold: (key -> 'a -> 'b -> 'b) -> 'a t -> 'b -> 'b
    (** [fold f m a] computes [(f kN dN ... (f k1 d1 a)...)],
        where [k1 ... kN] are the keys of all bindings in [m]
        (in increasing order), and [d1 ... dN] are the associated data. *)


  (** {6 Scanning} *)
    
  val find: key -> 'a t -> 'a
    (** [find x m] returns the current binding of [x] in [m],
        or raises [Not_found] if no such binding exists. *)
    
  val mem: key -> 'a t -> bool
    (** [mem x m] returns [true] if [m] contains a binding for [x],
        and [false] otherwise. *)
    
  val first : 'a t -> key * 'a
    (** Return the minimum key and its associated value, or [Not_found] if map is empty. *)
    
end
  (** Output signature of the functor {!Map.Make}. *)
  
module Make (Ord : OrderedType) : S with type key = Ord.t
  (** Functor building an implementation of the map structure given a totally ordered type. *)

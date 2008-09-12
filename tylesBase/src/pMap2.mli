(** Polymorphic Map.
    
    This is a polymorphic map, similar to standard library [Map] module
    but in a defunctorized style.

    Extension of ExtLib's PMap. 
*)

type ('a, 'b) t
    
val empty : ('a, 'b) t
  (** The empty map, using [compare] as key comparison function. *)
  
val is_empty : ('a, 'b) t -> bool
  (** returns true if the map is empty. *)
  
val size : ('a, 'b) t -> int
  (** Number of bindings in given map. *)
  
val create : ('a -> 'a -> int) -> ('a, 'b) t
  (** creates a new empty map, using the provided function for key comparison.*)
  
val add : 'a -> 'b -> ('a, 'b) t -> ('a, 'b) t
  (** [add x y m] returns a map containing the same bindings as
      [m], plus a binding of [x] to [y]. If [x] was already bound
      in [m], its previous binding disappears. *)
  
val find : 'a -> ('a, 'b) t -> 'b
  (** [find x m] returns the current binding of [x] in [m],
      or raises [Not_found] if no such binding exists. *)
  
val remove : 'a -> ('a, 'b) t -> ('a, 'b) t
  (** [remove x m] returns a map containing the same bindings as
      [m], except for [x] which is unbound in the returned map. *)
  
val mem : 'a -> ('a, 'b) t -> bool
  (** [mem x m] returns [true] if [m] contains a binding for [x],
      and [false] otherwise. *)
  
val exists : 'a -> ('a, 'b) t -> bool
  (** same as [mem]. *)
  
val iter : ('a -> 'b -> unit) -> ('a, 'b) t -> unit
  (** [iter f m] applies [f] to all bindings in map [m].
      [f] receives the key as first argument, and the associated value
      as second argument. The order in which the bindings are passed to
      [f] is unspecified. Only current bindings are presented to [f]:
      bindings hidden by more recent bindings are not passed to [f]. *)
  
val map : ('b -> 'c) -> ('a, 'b) t -> ('a, 'c) t
  (** [map f m] returns a map with same domain as [m], where the
      associated value [a] of all bindings of [m] has been
      replaced by the result of the application of [f] to [a].
      The order in which the associated values are passed to [f]
      is unspecified. *)
  
val mapi : ('a -> 'b -> 'c) -> ('a, 'b) t -> ('a, 'c) t
  (** Same as [map], but the function receives as arguments both the
      key and the associated value for each binding of the map. *)
  
val fold : ('b -> 'c -> 'c) -> ('a , 'b) t -> 'c -> 'c
  (** [fold f m a] computes [(f kN dN ... (f k1 d1 a)...)],
      where [k1 ... kN] are the keys of all bindings in [m],
      and [d1 ... dN] are the associated data.
      The order in which the bindings are presented to [f] is
      unspecified. *)
  
val foldi : ('a -> 'b -> 'c -> 'c) -> ('a , 'b) t -> 'c -> 'c
  (** Same as [fold], but the function receives as arguments both the
      key and the associated value for each binding of the map. *)
  
val enum : ('a, 'b) t -> ('a * 'b) Enum.t
  (** creates an enumeration for this map. *)
  
val of_enum : ?cmp:('a -> 'a -> int) -> ('a * 'b) Enum.t -> ('a, 'b) t
  (** creates a map from an enumeration, using the specified function
      for key comparison or [compare] by default. *)

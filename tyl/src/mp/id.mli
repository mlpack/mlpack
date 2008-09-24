type t

val make : string -> t
val equal : t -> t -> bool
val toString : t -> string

module Set : sig
  include Set.S
  val unions : t list -> t
  val addAll : elt list -> t -> t
end with type elt = t

val fresh : Set.t -> t -> t 
val fresh' : int -> Set.t -> t -> t list
val fresh'' : int -> Set.t -> t list -> t list list

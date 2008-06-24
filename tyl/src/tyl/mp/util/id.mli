type t

val make : string -> t
val equal : t -> t -> bool

module Set : sig
  include Set.S
  val union' : t list -> t
end with type elt = t

(* return an id not in set, using input id as root of returned it *)
val fresh : Set.t -> t -> t 


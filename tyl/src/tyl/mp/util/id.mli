type t

val equal : t -> t -> bool
module Set : Set.S with type elt = t  
val fresh : Set.t -> t -> t (* return an id not in set, using input id as root of returned it *)


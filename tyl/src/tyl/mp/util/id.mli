type t

val equal : t -> t -> bool
val fresh : Set.t -> t -> t (* return an id not in set, using input id as root of returned it *)

module Set : Set.S with type elt = t  

(*
val compare : t -> t -> int
val toString : t -> string
val shorter : t -> t -> t
val shortest : t list -> t option
val getMark : t -> Pos.range option
val concat : t list -> t
  
module Map : Map.S with type key = t

val freshl : Set.t -> t -> int -> t list (* return k id's not in given set *)
val freshll : Set.t -> t list -> int -> t list list (* return k id's for each given id *)
*)

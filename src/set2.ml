module type OrderedType = Set.OrderedType

module type S = sig
  include Set.S
  val of_list : elt list -> t
  val to_list : t -> elt list
end

module Make (Ord:OrderedType) = struct
  include (Set.Make(Ord) : Set.S with type elt = Ord.t)

  let of_list el = List.fold_left (fun ans e -> add e ans) empty el
  let to_list t = List.rev (fold (fun e ans -> e::ans) t [])
end

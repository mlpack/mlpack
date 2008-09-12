module Array = Array2
module List = List2

module type OrderedType = Map.OrderedType

module type S = sig
  type key
  type (+'a) t
  val is_empty : 'a t -> bool
  val size : 'a t -> int
  val compare : ('a -> 'a -> int) -> 'a t -> 'a t -> int
  val equal : ('a -> 'a -> bool) -> 'a t -> 'a t -> bool
  val empty : 'a t
  val add: key -> 'a -> 'a t -> 'a t
  val remove: key -> 'a t -> 'a t
  val of_array : (key * 'a) array -> 'a t
  val of_list : (key * 'a) list -> 'a t
  val to_array : 'a t -> (key * 'a) array
  val to_list : 'a t -> (key * 'a) list
  val iter: (key -> 'a -> unit) -> 'a t -> unit
  val map: ('a -> 'b) -> 'a t -> 'b t
  val map2: ('a -> 'b -> 'c) -> 'a t -> 'b t -> 'c t
  val mapi: (key -> 'a -> 'b) -> 'a t -> 'b t
  val map2i: (key -> 'a -> 'b -> 'c) -> 'a t -> 'b t -> 'c t
  val fold: (key -> 'a -> 'b -> 'b) -> 'a t -> 'b -> 'b
  val find: key -> 'a t -> 'a
  val mem: key -> 'a t -> bool
  val first : 'a t -> key * 'a
end

module Make (Ord:OrderedType) = struct
  include (Map.Make(Ord) : Map.S with type key = Ord.t)

  let size t = fold (fun _ _ ans -> ans + 1) t 0

  let of_array a = Array.fold_left (fun ans (k,a) -> add k a ans) empty a
  let of_list l = List.fold_left (fun ans (k,a) -> add k a ans) empty l
    
  let to_list t = List.rev (fold (fun k a ans -> (k,a)::ans) t [])
  let to_array t = Array.of_list (to_list t)

  let map2i f m n =
    if size m <> size n then failwith "domains not equal in size";
    let mn = List.zip (to_list m) (to_list n) in
    let f ans ((k1,a),(k2,b)) =
      if Ord.compare k1 k2 = 0
      then add k1 (f k1 a b) ans
      else failwith "domains contain different keys"
    in
      List.fold_left f empty mn

  let map2 f m n = map2i (fun _ a b -> f a b) m n      

  let first m =
    let ans = ref None in
    try
      iter (fun k x -> ans := Some(k,x); raise Exit) m;
      raise Not_found
    with 
        Exit -> match !ans with None -> raise Not_found | Some kx -> kx
      
end

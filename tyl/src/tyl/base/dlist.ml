open Util

type 'a dlist = 'a list -> 'a list

let empty     : 'a dlist             
  = id
let singleton : 'a -> 'a dlist       
  = cons
let append    : 'a dlist -> 'a dlist -> 'a dlist  
  = (%)

(* let foreach   : 'a dlist -> ('a -> 'b dlist) -> 'b dlist *)
(*   =  *)
let fromList = (@)
let toList xs = xs []

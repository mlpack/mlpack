type 'a t
type 'a series = unit -> 'a t
type 'a tester = int -> ('a * exn) option

val fold : 'a series -> int -> 'b -> ('b->'a->'b) -> 'b

val pure : 'a -> 'a series
val (%%) : ('a->'b) series -> 'a series -> 'b series
val (++) : 'a series -> 'a series -> 'a series

val ints     : int series
val floats   : float series
val units    : unit series
val bools    : bool series
val pairs    : 'a series -> 'b series -> ('a * 'b) series
val options  : 'a series -> ('a option) series
val lists    : 'a series -> ('a list) series

exception Counterexample_found
val forAll   : 'a series -> ('a -> bool) -> 'a tester

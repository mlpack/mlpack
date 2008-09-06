type 'a series = int -> 'a list

val ints     : int series
val floats   : float series
val units    : unit series
val bools    : bool series
val options  : 'a series -> ('a option) series
val lists    : 'a series -> ('a list) series

val (++) : 'a series -> 'a series -> 'a series
val (><) : 'a series -> 'b series -> ('a * 'b) series
val lift : ('a -> 'b) -> 'a series -> 'b series
val return : 'a -> 'a series

val evaluate : int -> (int -> 'a option) -> 'a option
val forAll : 'a series -> ('a -> bool) -> int -> 'a option

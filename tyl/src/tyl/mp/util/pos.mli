type t
type range

(*      
val pos : string -> int -> int -> t  (* file name, line num, char pos *)
val eof : string -> t
val range : t -> t -> range (* from one position to another *)
    
val left : range -> t
val right : range -> t
val union : range -> range -> range
val uniono : range option -> range option -> range option
    
val toString : t -> string
val toStringR : range -> string      
*)

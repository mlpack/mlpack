(** Array printing routines. Default behavior is:
    - printing is to [stdout], change by using optional [cout] argument
    - each item of a 1-D array is printed on a separate line, change by using optional [sep] argument
    - columns of a 2-D array are separated by tabs, change by using optional [sep] argument 
*)

val i : ?cout:out_channel -> ?sep:string -> int array -> unit
val f : ?cout:out_channel -> ?sep:string -> float array -> unit
val s : ?cout:out_channel -> ?sep:string -> string array -> unit
val a : ?cout:out_channel -> ?sep:string -> ('a -> string) -> 'a array -> unit
  
val ii : ?cout:out_channel -> ?sep:string -> int array array -> unit
val ff : ?cout:out_channel -> ?sep:string -> float array array -> unit
val ss : ?cout:out_channel -> ?sep:string -> string array array -> unit
val aa : ?cout:out_channel -> ?sep:string -> ('a -> string) -> 'a array array -> unit

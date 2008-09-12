(** Generally useful operations. *)

val ( @ ) : 'a list -> 'a list -> 'a list
  (** ExtLib's new append operator. *)
  
val identity : 'a -> 'a
  (** The identity function. *)
  
val (<<-) : ('b -> 'c) -> ('a -> 'b) -> ('a -> 'c)
  (** Function composition in normal direction as used in mathematics, [(f <<- g) x = f(g x)]. *)
  
val (->>) : ('a -> 'b) -> ('b -> 'c) -> ('a -> 'c)
  (** Function composition in reverse direction, [(f ->> g) x = g(f x)]. *)

val (&) : ('a -> 'b) -> 'a -> 'b
  (** Function application operator. Can be used to reduce number of parentheses. For example, can write [f & g x] instead of [f(g x)]. *)
  
val flip : ('a -> 'b -> 'c) -> ('b -> 'a -> 'c)
  (** [flip f] returns a function that takes its arguments in opposite order of [f]. *)

val open_out_safe : string -> out_channel
  (** Like [Pervasives.open_out] but raise [Sys_error] if file already exists. *)
  
val output_endline : out_channel -> string -> unit
  (** Write string on given output channel followed by a newline. The buffer is not necessarily flushed as in [print_endline] and [prerr_endline]. *)
  
val eps_float : float -> float
  (** [eps_float v] returns nearly the smallest (or perhaps the actual smallest) positive number [x] such that [v +. x <> v]. (Courtesty of Christophe Troestler and Mathias Kende, posted on OCaml Beginners List.) *)

val try_finally : ('a -> 'b) -> ('a -> unit) -> 'a -> 'b
  (** [try_finally f g x] returns [f x] after executing [g x]. If both [f] and [g] raise exceptions, it will be [f]'s exception that is raised by [try_finally]. Example: [try_finally input_line close_in (open_in "file.txt")] will read a line from "file.txt", assuring that opened channel is closed. (Courtesy of Jon Harrop, posted on OCaml Beginners List.) *)

val string_of_float : float -> string
  (** Like Standard Library's [string_of_float] but decimal value included even when 0, e.g. will generate "1.0" instead of "1.". *)

val print_float : float -> unit
  (** Like Standard Library's [print_float] but decimal value included even when 0, e.g. will generate "1.0" instead of "1.". *)

val float_of_stringi : string -> float
  (** [float_of_stringi s] returns a float if [s] represents either an int or float. *)

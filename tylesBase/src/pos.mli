(** File positions. *)

type t = private {
  file:string option; (** file name *)
  line:int option; (** line number *)
  col:int option; (** column number, can be defined only if line number is too *)
}
    
exception Bad of string

exception Undefined
  (** Raised when asking for undefined position information. *)
  
val f : string -> t
val l : int -> t
val fl : string -> int -> t
val lc : int -> int -> t
val flc : string -> int -> int -> t
  (** Methods for creating a position. [f] stands for file name, [l] for line number, and [c] for column number. The arguments required correspond to the function name. There is no [fc] nor [c] function because a line number is required when a column number is given. *)
  
val unknown : t
  (** Represents an unknown position. Use sparingly. *)
  
val file_exn : t -> string
val line_exn : t -> int
val col_exn : t -> int
  (** Return the file name, line number, or column number. Raise {!Undefined} if given position does not have requested information. *)
  
val set_file : t -> string -> t
val set_line : t -> int -> t
val set_col : t -> int -> t
  (** Set the file name, line number, or column number. Raise [Bad] if resulting [t] would be ill-formed. *)

val incrl : t -> int -> t
  (** [incrl pos k] increments the line number of [pos] by [k]. *)
  
val to_string : t -> string
  (** String representation of a position. Intended for human legibility, no particular format guaranteed. *)
  

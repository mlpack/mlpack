(** Consistent printing of errors, warnings, and bugs. An error is a user mistake that prevents continuing program execution, a warning is a milder problem that the program continues to execute through, and a bug is a mistake in the software. *)

val err : ?pos:Pos.t -> string -> string
val warn : ?pos:Pos.t -> string -> string
val bug : ?pos:Pos.t -> string -> string
  (** Create a string communicating an error, warning, or bug. First optional arugment is position where problem ocurred. Second argument is a string explaining the problem. *)
  
val print_err : ?pos:Pos.t -> string -> unit
val print_warn : ?pos:Pos.t -> string -> unit
val print_bug : ?pos:Pos.t -> string -> unit
  (** Print an error, warning, or bug. First optional arugment is position where problem ocurred. Second argument is a string explaining the problem. *)
  
val max_array_length_error : string

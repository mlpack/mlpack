(** Parsing and printing of line-oriented text formats. Input sources and output destinations can be files, channels, streams, or strings. *)

exception Error of (Pos.t * string)
  (** Raised when there is a parse error. *)
  
val copy_file : ?first:int -> ?last:int -> string -> string -> unit
  (** [copy_file ~first ~last in_file out_file] copies line numbers [first] through [last] (inclusive) of [in_file] to [out_file]. Omitting [first] starts copying from beginning of file, omitting [last] copies to end of file, and thus omitting both copies entire file. Okay for [first] to be non-positive or [last] to exceed number of lines in [in_file]; copying simply done from beginning or to end of file. If copying to end of file, only difference possible is that [out_file] will end with newline even if [in_file] does not. *)


(** {8 Iterators} *)

val fold_file : ?strict:bool -> ('a -> string -> 'a) -> 'a -> string -> 'a
  (** [fold_file ~strict f init file] accumulates the result of applying [f] to each line of [file]. Function [f] should raise [Failure] to indicate an error for the given line. If [strict] is true, the default, this will be caught and re-raised as [Error (p,m)], where [p] gives the position of the error. If [strict] is false, the error message will be printed to stdout and parsing continues. *)
  
val fold_channel : ?strict:bool -> ('a -> string -> 'a) -> 'a -> in_channel -> 'a 
val fold_string : ?strict:bool -> ('a -> string -> 'a) -> 'a -> string -> 'a 
val fold_stream : ?strict:bool -> ('a -> string -> 'a) -> 'a -> char Stream.t -> 'a 


(** {8 Parsers} *)

val of_file : (string -> 'a option) -> string -> 'a list
  (** [of_file f file] reads all lines from [file], parsing each with [f]. Function [f] is passed the full contents of each line not including the final '\n' character. It should return None to skip the line. In case of error, it should raise [Failure m], where [m] is a message explaining the error. This will be caught and re-raised as [Error (p,m)], where [p] gives the position of the error. *)

val of_channel : (string -> 'a option) -> in_channel -> 'a list
val of_string : (string -> 'a option) -> string -> 'a list
val of_stream : (string -> 'a option) -> char Stream.t -> 'a list
  (** Like [of_file] but read input from alternative sources. Also, if [Error] is raised, its position will not contain a file name. *)


(** {8 Printers} *)
  
val to_file : ('a -> string) -> string -> 'a list -> unit
  (** [to_file f file l] prints each item of [l] on a separate line to [file], using [f] to convert each to a string. Every line, including the last one, is terminated by '\n'. *)

val to_channel : ('a -> string) -> out_channel -> 'a list -> unit
val to_string : ('a -> string) -> 'a list -> string
val to_stream : ('a -> string) -> 'a list -> char Stream.t
  (** Like [to_file] but print output to alternative destinations. *)

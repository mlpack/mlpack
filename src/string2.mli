(** Strings. Extension of ExtLib's ExtString, which itself extends Standard Library's String. *)

type t = string
    (** An alias for the type of strings. *)
    
val length : string -> int
  (** Return the length (number of characters) of the given string. *)
  
val get : string -> int -> char
  (** [String.get s n] returns character number [n] in string [s].
      The first character is character number 0.
      The last character is character number [String.length s - 1].
      You can also write [s.[n]] instead of [String.get s n].
      
      Raise [Invalid_argument "index out of bounds"]
      if [n] is outside the range 0 to [(String.length s - 1)]. *)
  
val set : string -> int -> char -> unit
  (** [String.set s n c] modifies string [s] in place,
      replacing the character number [n] by [c].
      You can also write [s.[n] <- c] instead of [String.set s n c].
      Raise [Invalid_argument "index out of bounds"]
      if [n] is outside the range 0 to [(String.length s - 1)]. *)
  
val compare: t -> t -> int
  (** The comparison function for strings, with the same specification as
      {!Pervasives.compare}.  Along with the type [t], this function [compare]
      allows the module [String] to be passed as argument to the functors
      {!Set.Make} and {!Map.Make}. *)
  

(** {6 Constructors} *)

val create : int -> string
  (** [String.create n] returns a fresh string of length [n].
      The string initially contains arbitrary characters.
      Raise [Invalid_argument] if [n < 0] or [n > Sys.max_string_length].
  *)
  
val make : int -> char -> string
  (** [String.make n c] returns a fresh string of length [n],
      filled with the character [c].
      Raise [Invalid_argument] if [n < 0] or [n > ]{!Sys.max_string_length}.*)
  
val init : int -> (int -> char) -> string
  (** [init l f] returns the string of length [l] with the chars
      f 0 , f 1 , f 2 ... f (l-1). *)
  
val copy : string -> string
  (** Return a copy of the given string. *)
  
val to_index : string -> int -> string
  (** [get_to s n] returns the substring of [s] from first char to index [n] (inclusive). *)
  
val from_index : string -> int -> string
  (** [get_from s n] returns the substring of [s] from index [n] to final char (inclusive). *)
  

(** {6 Converters} *)
  
val sub : string -> int -> int -> string
  (** [String.sub s start len] returns a fresh string of length [len],
      containing the characters number [start] to [start + len - 1]
      of string [s].
      Raise [Invalid_argument] if [start] and [len] do not
      designate a valid substring of [s]; that is, if [start < 0],
      or [len < 0], or [start + len > ]{!String.length}[ s]. *)
  
val fill : string -> int -> int -> char -> unit
  (** [String.fill s start len c] modifies string [s] in place,
      replacing the characters number [start] to [start + len - 1]
      by [c].
      Raise [Invalid_argument] if [start] and [len] do not
      designate a valid substring of [s]. *)
  
val blit : string -> int -> string -> int -> int -> unit
  (** [String.blit src srcoff dst dstoff len] copies [len] characters
      from string [src], starting at character number [srcoff], to
      string [dst], starting at character number [dstoff]. It works
      correctly even if [src] and [dst] are the same string,
      and the source and destination chunks overlap.
      Raise [Invalid_argument] if [srcoff] and [len] do not
      designate a valid substring of [src], or if [dstoff] and [len]
      do not designate a valid substring of [dst]. *)
  
val concat : string -> string list -> string
  (** [String.concat sep sl] concatenates the list of strings [sl],
      inserting the separator string [sep] between each. *)
  
val escaped : string -> string
  (** Return a copy of the argument, with special characters
      represented by escape sequences, following the lexical
      conventions of Objective Caml.  If there is no special
      character in the argument, return the original string itself,
      not a copy. *)

val uppercase : string -> string
  (** Return a copy of the argument, with all lowercase letters
      translated to uppercase, including accented letters of the ISO
      Latin-1 (8859-1) character set. *)
  
val lowercase : string -> string
  (** Return a copy of the argument, with all uppercase letters
      translated to lowercase, including accented letters of the ISO
      Latin-1 (8859-1) character set. *)
  
val capitalize : string -> string
  (** Return a copy of the argument, with the first character set to uppercase. *)
  
val uncapitalize : string -> string
  (** Return a copy of the argument, with the first character set to lowercase. *)

val replace_chars : (char -> string) -> string -> string
  (** [replace_chars f s] returns a string where all chars [c] of [s] have been
      replaced by the string returned by [f c]. *)
  
val replace : str:string -> sub:string -> by:string -> bool * string
  (** [replace ~str ~sub ~by] returns a tuple constisting of a boolean
      and a string where the first occurrence of the string [sub]
      within [str] has been replaced by the string [by]. The boolean
      is true if a subtitution has taken place. *)
  
val strip : ?chars:string -> string -> string
  (** Returns the string without the chars if they are at the beginning or
      at the end of the string. By default chars are " \t\r\n". *)
  
val stripl : ?chars:string -> string -> string
  (** Returns the string without the chars if they are at the beginning of the string. By default chars are " \t\r\n". *)    
  
val stripr : ?chars:string -> string -> string
  (** Returns the string without the chars if they are at the end of the string. By default chars are " \t\r\n". *)

val strip_final_cr : string -> string
  (** Strip final carriage return if there is one. *)
  
val split : string -> string -> string * string
  (** [split s sep] splits the string [s] between the first
      occurrence of [sep].
      raises [Invalid_string] if the separator is not found. *)
  
val nsplit : string -> string -> string list
  (** [nsplit s sep] splits the string [s] into a list of strings
      which are separated by [sep]. *)
  
val join : string -> string list -> string
  (** Same as [concat] *)
  
val slice : ?first:int -> ?last:int -> string -> string
  (** [slice ?first ?last s] returns a "slice" of the string
      which corresponds to the characters [s.[first]],
      [s.[first+1]], ..., [s[last-1]]. Note that the character at
      index [last] is {b not} included! If [first] is omitted it
      defaults to the start of the string, i.e. index 0, and if
      [last] is omitted is defaults to point just past the end of
      [s], i.e. [length s].  Thus, [slice s] is equivalent to
      [copy s].
      
      Negative indexes are interpreted as counting from the end of
      the string. For example, [slice ~last:-2 s] will return the
      string [s], but without the last two characters.
      
      This function {b never} raises any exceptions. If the
      indexes are out of bounds they are automatically clipped.
  *)
  
val lchop : string -> string
  (** Returns the same string but without the first character.
      does nothing if the string is empty. *)
  
val rchop : string -> string
  (** Returns the same string but without the last character.
      does nothing if the string is empty. *)
  
val of_int : int -> string
  (** Returns the string representation of an int. *)
  
val of_float : float -> string
  (** Returns the string representation of an float. *)
  
val of_char : char -> string
  (** Returns a string containing one given character. *)
  
val to_int : string -> int
  (** Returns the integer represented by the given string or
      raises [Invalid_string] if the string does not represent an integer.*)
  
val to_float : string -> float
  (** Returns the float represented by the given string or
      raises Invalid_string if the string does not represent a float. *)
  
val enum : string -> char Enum.t
  (** Returns an enumeration of the characters of a string.*)
  
val of_enum : char Enum.t -> string
  (** Creates a string from a character enumeration. *)
  
val explode : string -> char list
  (** [explode s] returns the list of characters in the string [s]. *)
  
val implode : char list -> string
  (** [implode cs] returns a string resulting from concatenating
      the characters in the list [cs]. *)
  
  
(** {6 Iterators} *)
  
val iter : (char -> unit) -> string -> unit
  (** [String.iter f s] applies function [f] in turn to all
      the characters of [s].  It is equivalent to
      [f s.[0]; f s.[1]; ...; f s.[String.length s - 1]; ()]. *)
  
val map : (char -> char) -> string -> string
  (** [map f s] returns a string where all characters [c] in [s] have been
      replaced by [f c]. **)
  
val fold_left : ('a -> char -> 'a) -> 'a -> string -> 'a
  (** [fold_left f a s] is
      [f (... (f (f a s.[0]) s.[1]) ...) s.[n-1]] *)
  
val fold_right : (char -> 'a -> 'a) -> string -> 'a -> 'a
  (** [fold_right f s b] is
      [f s.[0] (f s.[1] (... (f s.[n-1] b) ...))] *)
  

(** {6 Scanning} *)    

val contains : string -> char -> bool
  (** [String.contains s c] tests if character [c]
      appears in the string [s]. *)
  
val contains_from : string -> int -> char -> bool
  (** [String.contains_from s start c] tests if character [c]
      appears in the substring of [s] starting from [start] to the end
      of [s].
      Raise [Invalid_argument] if [start] is not a valid index of [s]. *)
  
val rcontains_from : string -> int -> char -> bool
  (** [String.rcontains_from s stop c] tests if character [c]
      appears in the substring of [s] starting from the beginning
      of [s] to index [stop].
      Raise [Invalid_argument] if [stop] is not a valid index of [s]. *)
  
val count : (char -> bool) -> string -> int
  (** [count f s] returns the number of characters in [s] that [f] is true for. *)
  
val exists : string -> string -> bool
  (** [exists str sub] returns true if [sub] is a substring of [str] or
      false otherwise. *)
  
val exists' : (char -> bool) -> string -> bool
  (** [exists' f s] returns true if [f] is true for any of the chars in [s]. So named because [exists] unfortunately used by ExtLib with different meaning. *)
  
val for_all : (char -> bool) -> string -> bool
  (** [for_all f s] returns true if [f] is true for all chars in [s]. *)

val ends_with : string -> string -> bool
  (** [ends_with s x] returns true if the string [s] is ending with [x]. *)
  
val starts_with : string -> string -> bool
  (** [starts_with s x] return true if [s] is starting with [x]. *)
  

(** {6 Searching} *)

val index : string -> char -> int
  (** [String.index s c] returns the position of the leftmost
      occurrence of character [c] in string [s].
      Raise [Not_found] if [c] does not occur in [s]. *)
  
val rindex : string -> char -> int
  (** [String.rindex s c] returns the position of the rightmost
      occurrence of character [c] in string [s].
      Raise [Not_found] if [c] does not occur in [s]. *)
  
val index_from : string -> int -> char -> int
  (** Same as {!String.index}, but start
      searching at the character position given as second argument.
      [String.index s c] is equivalent to [String.index_from s 0 c].*)
  
val rindex_from : string -> int -> char -> int
  (** Same as {!String.rindex}, but start
      searching at the character position given as second argument.
      [String.rindex s c] is equivalent to
      [String.rindex_from s (String.length s - 1) c]. *)
  
val find : string -> string -> int
  (** [find s x] returns the starting index of the string [x]
      within the string [s] or raises [Invalid_string] if [x]
      is not a substring of [s]. *)
  

(**/**)
  
external unsafe_get : string -> int -> char = "%string_unsafe_get"
external unsafe_set : string -> int -> char -> unit = "%string_unsafe_set"
external unsafe_blit :
  string -> int -> string -> int -> int -> unit = "caml_blit_string" "noalloc"
external unsafe_fill :
  string -> int -> int -> char -> unit = "caml_fill_string" "noalloc"    


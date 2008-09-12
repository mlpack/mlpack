(** Characters. Designed with only ASCII character set in mind. Extension of Standard Library's Char. *)

type t = char
    (** An alias for the type of characters. *)
    
external code : char -> int = "%identity"
    (** Return the ASCII code of the argument. *)
    
val chr : int -> char
  (** Return the character with the given ASCII code.
      Raise [Invalid_argument "Char.chr"] if the argument is
      outside the range 0--255. *)
  
val escaped : char -> string
  (** Return a string representing the given character,
      with special characters escaped following the lexical conventions
      of Objective Caml. *)
  
val lowercase : char -> char
  (** Convert the given character to its equivalent lowercase character. *)
  
val uppercase : char -> char
  (** Convert the given character to its equivalent uppercase character. *)
  
val compare: t -> t -> int
  (** The comparison function for characters, with the same specification as
      {!Pervasives.compare}.  Along with the type [t], this function [compare]
      allows the module [Char] to be passed as argument to the functors
      {!Set.Make} and {!Map.Make}. *)
  
val is_digit : char -> bool
val is_hex_digit : char -> bool
val is_oct_digit : char -> bool
val is_lower : char -> bool
val is_upper : char -> bool
val is_letter : char -> bool
val is_alpha_num : char -> bool
val is_ascii : char -> bool
  
val is_space : char -> bool
  (** [is_space c] returns true if [c] is in the string " \t\r\n". *)
  
val to_string : char -> string
  
val to_int : char -> int
  (** [to_int c] returns int corresponding to [c]. Raise [Invalid_argument] if [c] is not a digit. Do not confuse this with [code]. *) 
  
val from_int : int -> char
  (** [from_int k] returns char corresponding to k. Raise [Invalid_argument] if k not between 0 and 9 (inclusive). Do not confuse this with [chr]. *)
  
(**/**)
  
external unsafe_chr : int -> char = "%identity"

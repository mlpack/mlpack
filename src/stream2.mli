(** Streams. Extension of Standard Library's Stream. *)

type 'a t = 'a Stream.t
    (** The type of streams holding values of type ['a]. *)

exception Failure
  (** Raised by parsers when none of the first components of the stream
      patterns is accepted. *)
  
exception Error of string
  (** Raised by parsers when the first component of a stream pattern is
      accepted, but one of the following components is rejected. *)
  
val peek : 'a t -> 'a option
  (** Return [Some] of "the first element" of the stream, or [None] if
      the stream is empty. *)
  
val junk : 'a t -> unit
  (** Remove the first element of the stream, possibly unfreezing
      it before. *)
  
val count : 'a t -> int
  (** Return the current count of the stream elements, i.e. the number
      of the stream elements discarded. *)
  
val npeek : int -> 'a t -> 'a list
  (** [npeek n] returns the list of the [n] first elements of
      the stream, or all its remaining elements if less than [n]
      elements are available. *)
  
  
(** {6 Constructors}
    
    Warning: these functions create streams with fast access; it is illegal
    to mix them with streams built with [[< >]]; would raise [Failure]
    when accessing such mixed streams.
*)

val from : (int -> 'a option) -> 'a t
  (** [Stream.from f] returns a stream built from the function [f].
      To create a new stream element, the function [f] is called with
      the current stream count. The user function [f] must return either
      [Some <value>] for a value or [None] to specify the end of the
      stream. *)
  
val of_list : 'a list -> 'a t
  (** Return the stream holding the elements of the list in the same
      order. *)
  
val of_string : string -> char t
  (** Return the stream of the characters of the string parameter. *)
  
val of_channel : in_channel -> char t
  (** Return the stream of the characters read from the input channel. *)

val lines_of_chars : char t -> string t
  (** Split input char stream on '\n' characters. *)

val lines_of_channel : in_channel -> string t
  (** [lines_of_channel cin] is equivalent to [lines_of_chars (of_channel cin)]. *)
  
  
(** {6 Converters} *)
  
val truncate : int -> 'a t -> 'a t
  (** [truncate k s] returns the same stream as [s] but with at most [k] items. *)
  
val to_array : 'a t -> 'a array
  (** Return stream elements in an array. *)
  
val to_list : 'a t -> 'a list
  (** Return stream elements in a list. *)

val sub_stream : ('a list -> 'b) -> ('a -> bool) -> 'a t -> 'b t
  (** [sub_stream mk stop sa] creates a stream [sb] composed of sub-items of [sa]. Function [mk] specifies how the composition is to be done and [stop] specifies when to stop including items. Items in [sa] are included until first item satisfying [stop]. These are passed to [mk]. But also, any other items of [sa] satisfying [stop] are junked until the first one not satisfying [stop] (otherwise [sa] would not advance on subsequent calls to [next sb]. Note that [count] of input stream will also change even if you directly consume only [sb]. *)
  
  
(** {6 Iterators} *)
  
val iter : ('a -> unit) -> 'a t -> unit
  (** [Stream.iter f s] scans the whole stream s, applying function [f]
      in turn to each stream element encountered. *)

val fold : ('b -> 'a -> 'b) -> 'b -> 'a t -> 'b
  (** Like [List.fold_left]. *)

val keep_while : ('a -> bool) -> 'a t -> 'a t
  (** [keep_while pred s] returns a stream [s'] whose final element is just before the first one in [s] not satisfying [pred]. *)
  
val keep_whilei : (int -> 'a -> bool) -> 'a t -> 'a t
  (** Like {!keep_while} but the predicate is also given the stream count. *)
  
val skip_while : ('a -> bool) -> 'a t -> unit
  (** [skip_while pred s] advances [s] to the first element not satisfying [pred]. *)
  
val skip_whilei : (int -> 'a -> bool) -> 'a t -> unit
  (** Like {!skip_while} but the predicate is also given the stream count. *)

val map : ('a -> 'b) -> 'a t -> 'b t
  (** Convert a stream of [a]'s into a stream of [b]'s. *)
  
  
(** {6 Scanning} *)    
  
val is_empty : 'a t -> bool
  (** True if stream is empty *)
  
  
(** {6 Predefined parsers} *)
  
val next : 'a t -> 'a
  (** Return the first element of the stream and remove it from the
      stream. Raise Stream.Failure if the stream is empty. *)
  
val empty : 'a t -> unit
  (** Return [()] if the stream is empty, else raise [Stream.Failure]. *)
  
val one : ('a -> bool) -> 'a t -> 'a
  (** [one f s] returns the first element of [s] if it satisfies [f], and increments the stream position. Raise [Failure] otherwise, and stream position unaltered. *)
  
val many : ('a -> bool) -> 'a t -> 'a list
  (** [many f s] returns as many elements of [s] as match [f] in succession. Stream position set to first element not satisfying [f]. Raise [Failure] if first element does not satisfy [f]. *)    

(**/**)
  
(** {6 For system use only, not for the casual user} *)
  
val iapp : 'a t -> 'a t -> 'a t
val icons : 'a -> 'a t -> 'a t
val ising : 'a -> 'a t
  
val lapp : (unit -> 'a t) -> 'a t -> 'a t
val lcons : (unit -> 'a) -> 'a t -> 'a t
val lsing : (unit -> 'a) -> 'a t
  
val sempty : 'a t
val slazy : (unit -> 'a t) -> 'a t
  
val dump : ('a -> unit) -> 'a t -> unit

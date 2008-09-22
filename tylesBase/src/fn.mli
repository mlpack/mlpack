(** Function applications. *)

val repeat : int -> ('a -> 'a) -> 'a -> 'a
  (** [repeat n f init] returns [f (f ... (f init) ... )], the function [f] applied [n] times to its own result, using [init] as the argument to the first call. Returns [init] if [n <= 0]. *)
  
val repeati : int -> (int -> 'a -> 'a) -> 'a -> 'a
  (** [repeati n f init] returns [f n (f (n-1) ... (f 1 init) ... )]. It is like [repeat] but [f] is also passed the application number. *)
  

(** Signature of types supporting an order relation. *)

module type S = sig  
  type t
      (** Type of items supporting an order relation. *)
      
  val compare : t -> t -> int
    (** The order relation on items of type [t]. *)
end

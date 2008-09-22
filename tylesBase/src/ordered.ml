module type S = sig
  type t
  val compare : t -> t -> int
end

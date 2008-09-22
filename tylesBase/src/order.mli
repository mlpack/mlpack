(** Operations on order relations. *)

val compose : ('a -> 'a -> int option) -> ('a -> 'a -> int option) -> ('a -> 'a -> int)
  (** [compose cmp1 cmp2] constructs a total order from two partial orders [cmp1] and [cmp2]. It does so by returning the result of [cmp1] if [cmp1] relates the items. If not, it returns the result of [cmp2].
      
      It is the caller's responsibility to assure that [cmp1] and [cmp2] have the appropriate properties to assure that the result will be a true total order. A good way to assure this is to make sure that for any two values [a] and [b], at most one of the following is true: [cmp1 a b = Some Less], [cmp1 a b = Some Greater], [cmp2 a b = Some Less], or [cmp2 a b = Some Greater]. If none of these is true, then both [cmp1 a b = Some Equal] and [cmp2 a b = Some Equal] should be true. I believe these are sufficient conditions, but not sure if necessary. *)
 
val reverse : ('a -> 'a -> int) -> ('a -> 'a -> int)
  (** [reverse cmp] reverses the ordering of [cmp]. *)
 
val reversep : ('a -> 'a -> int option) -> ('a -> 'a -> int option)
  (** [reversep cmp] reverses the ordering of a partial order [cmp]. *)

val totalify : ('a -> 'a -> int option) -> ('a -> 'a -> int)
  (** [totalify cmp] converts the partial order [cmp] into a total order. The resulting order relation will raise [Failure] if used on elements for which [cmp] would return [None]. *)

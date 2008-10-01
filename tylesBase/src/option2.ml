include Option

let map2 f x y = match x with None -> None | Some x' -> map (f x') y

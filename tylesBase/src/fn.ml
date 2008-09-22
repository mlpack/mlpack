let repeati n f init =
  let rec loop ans i =
    if i > n then ans else loop (f i ans) (i+1)
  in
    loop init 1

let repeat n f = repeati n (fun _ v -> f v)

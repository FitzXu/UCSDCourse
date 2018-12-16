let rev = foldl (fun acc -> fun nxt -> nxt :: acc) [] in
let m = [1;3;4] in
rev m
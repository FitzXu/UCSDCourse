let sumList = foldl (fun acc -> fun nxt -> acc + nxt) 0 in
let m = [1;3;4] in
sumList m
#### m != n:
Uncompressed size = mn\
Compressed size = mk + k + nk

=> Compression ratio = mn/(k(1 + m + n))\
Only worth using if CR > 1:\
=> k < mn/(1 + m + n)

#### m = n:
Compression ratio = m²/(k(1 + 2m))\
Only worth using if CR > 1:\
=> k < m²/(1 + 2m)

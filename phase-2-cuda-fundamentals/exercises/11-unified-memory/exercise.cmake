# Each variant runs a full round trip over 512 MB several times,
# and the oversubscription test allocates 1.5x device memory.
set(EX_TIMEOUT 600)

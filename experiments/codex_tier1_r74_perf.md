CLEAN

No material performance issues in `161e5a8`. The change only adds a status check around an `O(1)` `dict.pop()` in the checkpoint-resume path, which is not a hot loop and does not introduce meaningful CPU, I/O, or memory regression beyond pre-existing checkpoint load costs.
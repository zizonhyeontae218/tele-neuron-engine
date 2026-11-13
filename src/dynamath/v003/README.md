# DynaMath 003 implementation

A bounded serializable elementwise AST evolves one of the nine v002 recurrent laws. `Model` adds the interpreter and retains Model1 scalar dynamics; `Session` is the unchanged v002 FIFO and clock. Run `python -m dynamath.v003 CHECKPOINT` for corpus-free inference. See `docs/dynamath/experiment-003/README.md` for protocol and limits.

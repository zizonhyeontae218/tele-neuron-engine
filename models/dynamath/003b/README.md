# DynaMath 003b checkpoint storage

The production runner creates `latest.pt` atomically and milestone checkpoints
`step-05000.pt`, `step-20000.pt`, `step-35000.pt`, `step-50000.pt`.
These large binaries are ignored by Git; source history contains no repeated
model blobs. Actions artifacts `dynamath-003b-checkpoints-<segment>` retain them
for 90 days. Download and archive them before expiry. Evidence artifacts include
the checkpoint hashes and tokenizer/data hashes needed to check identity.

Every checkpoint holds weights/config/ASTs plus optimizer, LR schedule, Python,
NumPy, Torch and CUDA RNG (when applicable), completed update, lane positions and
epochs, S/m and per-lane clocks, contract/data/tokenizer hashes and environment.
Resume is exact on the tested CPU environment. Different Torch/CUDA versions or
hardware can change floating-point numerics; checkpoint provenance records both.

Only load trusted `.pt` files: these checkpoints include Python RNG/optimizer
objects and therefore require PyTorch's `weights_only=False` deserialization.

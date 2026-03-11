# Contributing

Thanks for your interest in contributing! See the [README](README.md) for setup and build instructions.

## Pull requests

- Make sure CI passes. Warnings in `src/` are treated as errors.
- Describe **what** the change does and **why** in the PR description.
- Keep changes focused. One logical change per PR.

## Code style

- C++20. Use modern idioms (RAII, smart pointers, `std::filesystem`, `enum class`).
- No raw `new`/`delete`/`malloc`/`free`. Use containers and smart pointers.
- Follow the patterns in existing code. When in doubt, keep it simple.
- Vendor code under `vendor/` is not ours to modify. Report issues upstream.

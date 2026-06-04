"""
Sanity check for the dlens library.

Run with:
    uv run python scripts/sanity_check.py
"""

import sys


def main() -> None:
    print("Python executable:")
    print(sys.executable)
    print()

    try:
        import dlens
        from dlens.core import ping
    except ImportError as e:
        print("Import failed")
        raise

    print("Import succeeded")
    print(f"dlens module: {dlens}")
    print(f"ping(): {ping()}")

    assert ping() == "dlens ok", "Unexpected ping() result"

    print("\nSanity check passed")


if __name__ == "__main__":
    main()

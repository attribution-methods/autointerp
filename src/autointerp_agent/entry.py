"""Featherweight console-script entry point.

The real app pulls in rich / pydantic / litellm, which takes a second or
two; a Ctrl-C during that window would otherwise escape as a raw
KeyboardInterrupt traceback. This module imports nothing heavy at module
level so the unguarded window is just interpreter startup.
"""

from __future__ import annotations


def main() -> int:
    try:
        from autointerp_agent.app import main as app_main

        return app_main()
    except KeyboardInterrupt:
        print()
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

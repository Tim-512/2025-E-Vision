from __future__ import annotations

import json

from ev_vision.camera.hikrobot import create_native_api


def main() -> int:
    api = create_native_api()
    print(json.dumps(api.list_devices(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

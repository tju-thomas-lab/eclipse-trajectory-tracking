from __future__ import annotations

import json
import sys
from pathlib import Path

from eclipse_trajectory.ingest.video import inspect_video

if __name__ == "__main__":
    metadata = inspect_video(Path(sys.argv[1]))
    print(json.dumps(metadata.model_dump(mode="json"), indent=2, sort_keys=True))

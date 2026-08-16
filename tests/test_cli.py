from __future__ import annotations

import json

from audio_highlight.cli import run


def test_validate_segments_command(tmp_path, capsys) -> None:
    path = tmp_path / "segments.json"
    path.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "start_frame": 0,
                        "end_frame": 90,
                        "start_sec": 0.0,
                        "end_sec": 3.0,
                        "duration_sec": 3.0,
                    }
                ],
                "fps": 30.0,
            }
        ),
        encoding="utf-8",
    )

    assert run(["validate-segments", str(path)]) == 0
    assert "1 segments, fps=30" in capsys.readouterr().out

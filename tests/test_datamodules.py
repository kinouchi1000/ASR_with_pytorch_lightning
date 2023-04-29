import shutil
from pathlib import Path

from src.data.formatter.jsut import JSUTFormatter


def test_format_json():
    jsut = JSUTFormatter("./tmp")
    manifest_path = jsut.get_manifest_path()
    assert Path(manifest_path).exists()
    manifest = jsut.get_manifest()
    assert len(manifest) > 0
    for item in manifest:
        assert "text" in item
        assert "wav_path" in item
        assert "id" in item
        assert "speaker_id" in item
    shutil.rmtree("./tmp")

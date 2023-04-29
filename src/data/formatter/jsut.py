import json
import shutil
from glob import glob
from pathlib import Path

import wget
from loguru import logger


class JSUTFormatter:

    """
    Download and format JSUT corpus
    Of course, you can use jsut module https://github.com/tarepan/jsut
    """

    def __init__(self, data_path: str):
        JSUT_URL = "http://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip"
        self.save_path = Path(data_path) / "JSUT"
        self.save_path.mkdir(parents=True, exist_ok=True)

        if not (self.save_path / "jsut_ver1.1").exists():
            wget.download(JSUT_URL, str(self.save_path) + "/jsut_ver1.1.zip")
            shutil.unpack_archive(str(self.save_path) + "/jsut_ver1.1.zip", str(self.save_path))

        self.jsut_dir_path = self.save_path / "jsut_ver1.1"
        self._make_manifest(self.jsut_dir_path)

    def _make_manifest(self, path_list: Path):
        logger.info("START: Make manifest File")
        courpus_list = glob(str(path_list) + "/**/")
        all_manifest = []
        for courpus in courpus_list:
            courpus_name = courpus.split("/")[-2]
            logger.info(f"courpus {courpus_name}")

            # make transcripts
            transcript_file = courpus + "/transcript_utf8.txt"
            transcripts = self._transcript_to_dict(transcript_file)

            # get wav files
            for wav_path in glob(courpus + "/wav/*.wav"):
                wav_id = wav_path.split("/")[-1].split(".")[0]

                text = transcripts[wav_id]
                manifest = {"id": wav_id, "wav_path": wav_path, "text": text, "speaker_id": "jsut"}
                all_manifest.append(manifest)

        with open(self.save_path / "all_manifest.json", "w") as f:
            json.dump(all_manifest, f, indent=4)

    def _transcript_to_dict(self, filename: str) -> dict:
        with open(filename) as f:
            transcript_dict = {}
            lines = f.readlines()
        for line in lines:
            id, text = line.strip().split(":")
            transcript_dict[id] = text
        return transcript_dict

    def get_manifest(self) -> dict:
        """
        get manifest dict from cache file
        """
        with open(self.save_path / "all_manifest.json") as f:
            manifest = json.load(f)
        return manifest

    def get_manifest_path(self) -> str:
        """
        get manifest path from cache file
        """
        return str(self.save_path / "all_manifest.json")

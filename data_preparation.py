import torch
from pathlib import Path

DATASET_NAME = "chapter1"
DATA_DIR = Path(__file__).resolve().parent


def prepare_data(text_path, output_path):
	"""
	Reads a text file, builds the alphabet, and writes a single .pt file
	containing:
	  - "alphabet": sorted list of unique lowercase chars
	  - "samples":  list of LongTensors, one per non-empty line,
	                each holding the alphabet index of every char in that line.
	"""
	with open(text_path, "r", encoding="utf-8") as f:
		lines = [line.lower() for line in f if len(line) > 1]

	alphabet = sorted(set("".join(lines)))
	char_to_idx = {c: i for i, c in enumerate(alphabet)}

	samples = [
		torch.tensor([char_to_idx[c] for c in line], dtype=torch.long)
		for line in lines
	]

	output_path.parent.mkdir(parents=True, exist_ok=True)
	torch.save({"alphabet": alphabet, "samples": samples}, output_path)
	print(f"Saved {len(samples)} samples, alphabet of {len(alphabet)} chars, "
	      f"to {output_path}")


if __name__ == "__main__":
	prepare_data(DATA_DIR / "tiny_tolkien.txt",
	             DATA_DIR / DATASET_NAME / "data.pt")

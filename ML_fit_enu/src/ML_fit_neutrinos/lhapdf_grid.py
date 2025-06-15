import torch


def write_lhapdf_grid(xgrid: torch.tensor, pdfprediction: torch.tensor, path: str):
    # xgrid = xgrid.numpy()
    pid = 12
    with open(path, "w") as f:
        f.write("PdfType: mean pdf\n")
        f.write("Format: lhagrid1\n")
        f.write("---\n")
        for val in xgrid:
            f.write(str(val.item()) + " ")
        f.write("\n")
        f.write("0.1E+001 0.1E+007\n")
        f.write(f"{pid}\n")
        for val in pdfprediction:
            f.write(f"{val.item()}\n")
            f.write(f"{val.item()}\n")
        f.write("---")

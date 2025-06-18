import torch
import lhapdf


def write_lhapdf_grid(xgrid, pdfprediction, path, pid):
    # xgrid = xgrid.numpy()
    with open(path, "w") as f:
        f.write("PdfType: replica\n")
        f.write("Format: lhagrid1\n")
        f.write("---\n")
        for val in xgrid:
            f.write("  " + str(val.item()))
        f.write("\n")
        f.write("0.1E+001 0.1E+007\n")
        f.write(f"{pid}\n")
        for val in pdfprediction:
            f.write(f"  {val.item()}\n")
            f.write(f"  {val.item()}\n")
        f.write("---")


def customize_info_file(template_path, output_path, set_index, flavor):
    with open(template_path, "r") as file:
        content = file.read()

    # Replace placeholders
    content = content.replace("SETINDEX", str(set_index))
    content = content.replace("[FLAVOR]", str(flavor))

    # Write the customized content to a new file
    with open(output_path, "w") as file:
        file.write(content)

import torch
import lhapdf


def write_lhapdf_grid(xgrid, pdf_dict, path):
    with open(path, "w") as f:
        f.write("PdfType: replica\n")
        f.write("Format: lhagrid1\n")
        f.write("---\n")

        # Write x-grid values
        f.write("  " + "  ".join(f"{val:.8e}" for val in xgrid) + "\n")

        # Q² range (dummy or real)
        f.write("0.1E+001 0.1E+007\n")

        # Write PIDs in desired order
        pids = sorted(pdf_dict.keys())  # Optional: control order
        f.write(" ".join(str(pid) for pid in pids) + "\n")

        # Now for each x-point, write two lines:
        num_x = len(xgrid)
        for i in range(num_x):
            line = " ".join(f"{pdf_dict[pid][i]:.14e}" for pid in pids)
            f.write(line + "\n")
            f.write(line + "\n")  # LHAPDF format: duplicate each x-entry
        f.write("---\n")


def customize_info_file(template_path, output_path, set_index, flavor, num_members):
    with open(template_path, "r") as file:
        content = file.read()

    # Replace placeholders
    content = content.replace("SETINDEX", str(set_index))
    content = content.replace("FLAVOR", str(flavor))
    content = content.replace("NumMembers: 1000", f"NumMembers: {str(num_members)}")

    # Write the customized content to a new file
    with open(output_path, "w") as file:
        file.write(content)

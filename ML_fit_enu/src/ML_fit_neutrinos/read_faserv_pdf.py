import lhapdf
import numpy as np


def read_pdf(pdf, x_vals, particle):
    pid = particle
    Q2 = 10
    pdf = lhapdf.mkPDF(pdf, 2)
    pdf_vals = [pdf.xfxQ2(pid, x, Q2) for x in x_vals]
    pdf_vals = np.array(pdf_vals)
    pdf_vals /= x_vals
    return pdf_vals, x_vals


# pid = 12
# Q2 = 2
# pdf = lhapdf.mkPDF("faserv", 0)
# data, Enu, _, _, _ = read_hist()

# lowx = -5
# n = 50
# incexp = lowx / n
# x_vals = []
# for i in range(n):
#     ri = i
#     x_vals.append(np.exp(lowx - ri * incexp))

# _, fk_table = get_fk_table("FK_Enu.dat", -5, 50)
# pdf_vals = [pdf.xfxQ2(pid, x, Q2) for x in x_vals]
# pdf_vals = np.array(pdf_vals)
# pdf_vals /= x_vals
# flux = np.matmul(fk_table, pdf_vals)

# plt.plot(Enu, flux)
# plt.show()

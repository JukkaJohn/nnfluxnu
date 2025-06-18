import lhapdf
import numpy as np
import matplotlib.pyplot as plt


def read_pdf(pdf, x_vals, particle, set):
    pid = particle
    Q2 = 10
    pdf = lhapdf.mkPDF(pdf, set)
    pdf_vals = [pdf.xfxQ2(pid, x, Q2) for x in x_vals]
    pdf_vals = np.array(pdf_vals)
    pdf_vals /= x_vals
    return pdf_vals, x_vals


# pdf_vals, x_vals = read_pdf("testgrid", np.arange(0.01, 0.98, 0.001), 12, 1)
# print(pdf_vals)
# plt.plot(x_vals, pdf_vals)
# plt.yscale("log")
# plt.xscale("log")
# plt.show()
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

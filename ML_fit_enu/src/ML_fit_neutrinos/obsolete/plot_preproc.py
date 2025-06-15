import matplotlib.pyplot as plt
from read_faserv_pdf import read_pdf
from logspace_grid import generate_grid
import numpy as np

alpha, beta, gamma = -2.6, 100, 10000000
# alpha = 3.4
# beta = 8.3
# gamma = 34
# num_obs = 1

# x = np.logspace(-8, 0, 1000)
pdf = "faserv"
lowx = -8
n = 250
x_vals = generate_grid(lowx, n)
x_vals = np.array(x_vals)
y = gamma * (1 - x_vals) ** beta * x_vals ** (1 - alpha)


faser_pdf, x_faser = read_pdf(pdf, x_vals)
plt.plot(x_faser, faser_pdf * x_vals, label="faserv pdf")
plt.plot(
    x_vals,
    y / x_vals,
    label=r"$y = A\cdot x^{1-\alpha}\cdot (1-x)^{\beta}$",
)

# alpha, beta, gamma, epsilon, a = 0.7, 30, 40000, -800, 0.6
# y = a * x ** (-alpha) * (1 - x) ** beta * (1 + epsilon * x**0.5 + gamma * x)
# plt.plot(
#     x,
#     y,
#     label=r"$y = A\cdot x^{-\alpha}\cdot (1-x)^{\beta}\cdot(1+\epsilon\cdot x^{0.5}+\gamma\cdot x)$",
# )
plt.xlabel(r"$x_{\nu_i}$", fontsize=16)
plt.ylabel(r"$f_{\nu_i}(x_{\nu})$", fontsize=16)
plt.ylim(10**-8, 10**5)
# plt.xlim(10**-4, 1.1)
plt.xscale("log")
plt.yscale("log")
# plt.title(f"alpha ={alpha},beta = {beta} gamma = {gamma}")
plt.legend()
# plt.savefig("preproc.pdf")
plt.show()

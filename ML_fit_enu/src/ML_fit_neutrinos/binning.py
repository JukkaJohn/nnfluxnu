def return_binning(bpd, plow, phigh):
    maxbins = 400
    inc = 1 / bpd
    xx = []
    xx.append(10**plow)

    binsizes, high_bin, low_bin = [], [], []
    for i in range(2, maxbins + 1):
        xx.append(10 ** (plow + inc * (i - 1)))
        if plow + inc * (i - 1) - phigh > 0:
            break
        binsizes.append(xx[i - 1] - xx[i - 2])
        high_bin.append(xx[i - 1])
        low_bin.append(xx[i - 2])
    return high_bin, low_bin, binsizes


high_bin, low_bin, binsizes = return_binning(50, 0, 4)
print(len(binsizes))
print(len(high_bin))
print(len(low_bin))
binsizes.append(4.7128548051e02)
low_bin.append(1.0000000000e04)
high_bin.append(1.0471285481e04)

filename = "FK_El_fine_binsize.dat"
with open(filename, "w") as f:
    for i in range(len(binsizes)):
        f.write(f"{low_bin[i]:.10e} {high_bin[i]:.10e} {binsizes[i]:.10e}\n")

filename = "FK_Eh_fine_binsize.dat"
with open(filename, "w") as f:
    for i in range(len(binsizes)):
        f.write(f"{low_bin[i]:.10e} {high_bin[i]:.10e} {binsizes[i]:.10e}\n")

filename = "FK_Enu_fine_binsize.dat"
with open(filename, "w") as f:
    for i in range(len(binsizes)):
        f.write(f"{low_bin[i]:.10e} {high_bin[i]:.10e} {binsizes[i]:.10e}\n")

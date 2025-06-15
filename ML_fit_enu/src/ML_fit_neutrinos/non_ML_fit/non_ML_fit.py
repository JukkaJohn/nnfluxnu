from ML_fit_enu.src.ML_fit_neutrinos.non_ML_fit.nonMLnonFK_faser import perform_fit
from ML_fit_enu.src.ML_fit_neutrinos.non_ML_fit.FK_non_ML_fit import perform_fit_fk
import matplotlib.pyplot as plt


(
    rawx,
    fit,
    f_err_68,
    f_err_95,
    f_err_99,
    x,
    event_fit,
    pred,
    f_err_68_cont,
    f_err_95_cont,
    f_err_99_cont,
) = perform_fit_fk()


plt.grid(axis="both")
plt.plot(x, event_fit, drawstyle="steps-post", color="red", label="fk table fit")
plt.plot(x, pred, drawstyle="steps-post", color="blue", alpha=0.5, label="pred")
plt.xlabel(r"$x_{\nu}$", fontsize=16)
plt.ylabel(r"$f(x_{\nu})$", fontsize=16)
plt.grid(axis="both")
plt.legend()

plt.fill_between(
    x,
    (f_err_68 + event_fit),
    (-f_err_68 + event_fit),
    color="blue",
    step="post",
    alpha=0.5,
    label=r"1$\sigma$ fk fit",
)

plt.xlabel(r"$x_{\nu}$", fontsize=16)
plt.ylabel(r"$f(x_{\nu})$", fontsize=16)
plt.grid(axis="both")
plt.legend()


x, fit, f_ref, norm, f_err_68, f_err_95, f_err_99, p_min, pred = perform_fit()
plt.grid(axis="both")
plt.plot(x, event_fit, color="red", label="fit non fk")

plt.xlabel(r"$x_{\nu}$", fontsize=16)
plt.ylabel(r"$f(x_{\nu})$", fontsize=16)
plt.grid(axis="both")
plt.legend()
plt.title("events")
plt.fill_between(
    x,
    (f_err_68 + fit),
    (-f_err_68 + fit),
    color="orange",
    alpha=0.5,
    label=r"1$\sigma$ fit",
)

print("test")
plt.xlabel(r"$x_{\nu}$", fontsize=16)
plt.ylabel(r"$f(x_{\nu})$", fontsize=16)
plt.grid(axis="both")
plt.legend()
plt.show()

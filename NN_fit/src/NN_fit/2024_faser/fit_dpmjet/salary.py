total_sal = 23.8


x = 7
sal = x + x**1.035 + ((x**1.035) ** 1.035)

print((x + x**1.035 + ((x**1.035) ** 1.035)))
print((x + x**1.035 + ((x**1.035) ** 1.035)) / total_sal)
while total_sal / sal > 1:
    # print("yes")
    sal = x + x**1.035 + ((x**1.035) ** 1.035)
    # print(total_sal - sal)
    x += 0.001
    # print(x)
print(x)
print(x / 12)
print((x + x**1.035 + ((x**1.035) ** 1.035)))
print((x**1.035) / 12)
print(((x**1.035) ** 1.035) / 12)

x = 7.56
print((x + x**1.035 + ((x**1.035) ** 1.035)))

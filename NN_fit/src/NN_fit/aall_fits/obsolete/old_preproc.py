# if preproc == 0:
#     class CustomPreprocessing(nn.Module):
#         def __init__(self, alpha):
#             super(CustomPreprocessing, self).__init__()
#             self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32, requires_grad=True))
#         def forward(self, x):
#             return self.alpha*x

#     class PreprocessedMLP(nn.Module):
#         def __init__(self, alpha,l1, l2, l3):
#             super(PreprocessedMLP, self).__init__()
#             self.preprocessing = CustomPreprocessing(alpha)
#             self.mlp = SimplePerceptron(l1, l2, l3)

#         def forward(self, x):
#             x = self.preprocessing(x)
#             x = self.mlp(x)
#             return x

# if preproc == 2:
#     class CustomPreprocessing(nn.Module):
#         def __init__(self, alpha,beta,gamma,a,epsilon):
#             super(CustomPreprocessing, self).__init__()
#             self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32, requires_grad=True))
#             self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32, requires_grad=True))
#             self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32, requires_grad=True))
#             self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32, requires_grad=True))
#             self.epsilon = nn.Parameter(torch.tensor(epsilon, dtype=torch.float32, requires_grad=True))

#         def forward(self, x):
#             return self.a * x**(-self.alpha) * (1-x)**self.beta * (1+self.epsilon*x**0.5 + self.gamma*x)

#     class PreprocessedMLP(nn.Module):
#         def __init__(self, alpha, beta, gamma,a,epsilon,l1, l2, l3):
#             super(PreprocessedMLP, self).__init__()
#             self.preprocessing = CustomPreprocessing(alpha, beta,gamma,a,epsilon)
#             self.mlp = SimplePerceptron(l1, l2, l3)

#         def forward(self, x):
#             x = self.preprocessing(x)
#             x = self.mlp(x)
#             return x
# else:
#     print('else')
#     class CustomPreprocessing(nn.Module):
#         def __init__(self, alpha):
#             super(CustomPreprocessing, self).__init__()
#             self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32, requires_grad=True))
#         def forward(self, x):
#             return self.alpha*x

#     class PreprocessedMLP(nn.Module):
#         def __init__(self, alpha,l1, l2, l3):
#             super(PreprocessedMLP, self).__init__()
#             self.preprocessing = CustomPreprocessing(alpha)
#             self.mlp = SimplePerceptron(l1, l2, l3)

#         def forward(self, x):
#             x = self.preprocessing(x)
#             x = self.mlp(x)
#             return x


#  if preproc == 0:
#             alpha = 10
#             model = PreprocessedMLP(alpha,l1, l2, l3)


# if preproc == 2:
#             # alpha,beta,gamma,epsilon, a= -1.5,26,1,2000,2000
#             alpha, beta, gamma, epsilon, a = 0.7, 30, 40000, -800, 0.6
#             model = PreprocessedMLP(alpha, beta, gamma,a,epsilon,l1, l2, l3)

# else:
#     alpha = 10
#     model = PreprocessedMLP(alpha,l1, l2, l3)

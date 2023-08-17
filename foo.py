import torch.nn as nn
import torch

class Foo(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x):
        return self.linear(x)

# Instantiate the model
model = Foo()

# Train the model
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

input = torch.randn(1, 3)
loss = torch.sum(model(input) ** 2)
print(float(loss))

for _ in range(3):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss = torch.sum(model(input) ** 2)
    print(float(loss))

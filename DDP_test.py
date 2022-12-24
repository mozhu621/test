import torch
import torch.nn.functional as F
from datasets import load_dataset
from accelerate import Accelerator


accelerator = Accelerator()

model = torch.nn.Transformer()
optimizer = torch.optim.Adam(model.parameters())

dataset = load_dataset('my_dataset')
print(dataset)
data = torch.utils.data.DataLoader(dataset, shuffle=True)

model, optimizer, data = accelerator.prepare(model, optimizer, data)

model.train()
for epoch in range(10):
  for source, targets in data:

      optimizer.zero_grad()

      output = model(source)
      loss = F.cross_entropy(output, targets)


      accelerator.backward(loss)

      optimizer.step()
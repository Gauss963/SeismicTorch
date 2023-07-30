import torch

model = torch.load('./model/EarthquakeCNN_finetuned.pth')
model.eval()

example_input = torch.rand(1, 1, 80, 3)
traced_model = torch.jit.trace(model, example_input)

traced_model.save('./model/EarthquakeCNN_TS.pt')
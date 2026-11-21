import torch

EarthquakeCNN_model = torch.load('./model/EarthquakeCNN_finetuned.pth')
EarthquakeCNN_model.eval()

example_input = torch.rand(1, 1, 80, 3)
traced_model = torch.jit.trace(EarthquakeCNN_model, example_input)

traced_model.save('./model/EarthquakeCNN_TS.pt')



SpectrogramCNN_model = torch.load('./model/SpectrogramCNN_trained.pth')
SpectrogramCNN_model.eval()

example_input = torch.randn(1, 3, 100, 150)
traced_model = torch.jit.trace(SpectrogramCNN_model, example_input)

traced_model.save('./model/SpectrogramCNN_model_TS.pt')
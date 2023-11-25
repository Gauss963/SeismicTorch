import os
import torch
import numpy as np

def do_Spectrogram_dataprep_merge():
    
    datadir = './Spectrogram_data_pth'

    train_data_path = os.path.join(datadir, 'Spectrogram_event.pth')
    train_data = torch.load(train_data_path)
    Xevent = train_data['Xevent']
    Yevent = train_data['Yevent']

    print(Xevent.shape)
    print(Yevent.shape)

    train_data_path = os.path.join(datadir, 'Spectrogram_noise.pth')
    train_data = torch.load(train_data_path)
    Xnoise = train_data['Xnoise']
    Ynoise = train_data['Ynoise']

    print(Xnoise.shape)
    print(Ynoise.shape)

    Xall = np.concatenate((Xevent, Xnoise), axis = 0)
    Yall = np.concatenate((Yevent, Ynoise), axis = 0)

    print(Xall.shape)
    print(Yall.shape)


    torch.save({'Xall': Xall, 'Yall': Yall}, './Spectrogram_data_pth/Spectrogram_all_merged.pth')


if __name__ == "__main__":
    do_Spectrogram_dataprep_merge()
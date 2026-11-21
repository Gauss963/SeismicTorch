import os
import glob
import numpy as np
import obspy
import torch
import tqdm

import functions


def do_Spectrogram_dataprep_noise():

    stream_list = []
    Spectrogram_train_data_X = np.array([])
    Spectrogram_train_data_list = []

    file_name_part = 'HLX'

    datadir = './data_QSIS_Noise'
    file_name_part = 'HLX'

    sac_files_X = glob.glob(f'{datadir}/*{file_name_part}*.sac')
    sac_files_Y = [s.replace('X', 'Y') for s in sac_files_X]
    sac_files_Z = [s.replace('X', 'Z') for s in sac_files_X]


    for i in tqdm.trange(len(sac_files_X)):
        stream = obspy.Stream()
        st = obspy.read(sac_files_X[i])
        stream += st

        st = obspy.read(sac_files_Y[i])
        stream += st

        st = obspy.read(sac_files_Z[i])
        stream += st

        stream_list.append(stream)

    for i in tqdm.trange(len(stream_list)):
    # for i in tqdm.trange(4):

        Spectrogram_train_data_append = functions.stream_to_spectrogram_ndarray(stream_list[i])
        Spectrogram_train_data_list.append(Spectrogram_train_data_append)

    Spectrogram_train_data_X = np.stack(Spectrogram_train_data_list)
    Spectrogram_train_data_X = np.expand_dims(Spectrogram_train_data_X, axis = -1)


    initial_array = np.array([[0., 1.]])
    Spectrogram_train_data_Y = np.tile(initial_array, (len(stream_list), 1))

    print(Spectrogram_train_data_X.shape)
    print(Spectrogram_train_data_Y.shape)

    torch.save({'Xnoise': Spectrogram_train_data_X, 'Ynoise': Spectrogram_train_data_Y}, './Spectrogram_data_pth/Spectrogram_noise.pth')

    datadir = './Spectrogram_data_pth'

    train_data_path = os.path.join(datadir, 'Spectrogram_noise.pth')
    train_data = torch.load(train_data_path)
    Xevent = train_data['Xnoise']
    Yevent = train_data['Ynoise']

    print(Xevent.shape)
    print(Yevent.shape)




if __name__ == "__main__":
    do_Spectrogram_dataprep_noise()
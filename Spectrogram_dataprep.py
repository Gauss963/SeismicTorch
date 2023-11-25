import Spectrogram_dataprep_event
import Spectrogram_dataprep_noise
import Spectrogram_dataprep_merge


def do_Spectrogram_dataprep():
    Spectrogram_dataprep_event.do_Spectrogram_dataprep_event()
    Spectrogram_dataprep_noise.do_Spectrogram_dataprep_noise()
    Spectrogram_dataprep_merge.do_Spectrogram_dataprep_merge()


if __name__ == "__main__":
    do_Spectrogram_dataprep()
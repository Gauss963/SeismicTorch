# SeismicTorch
Seismic Event Classification using Convolutional Neutral Network applied to time series data.

This machine learning application to seismology uses a CNN to classify seismic events into earthquakes, active seismic sources and noise based on raw waveform records with a duration of 480 seconds using MEMS.

Package Requirements:
The pre-processing of the seismic data requires the Obspy package which is easy to install with pip.
1) Numpy
2) Scipy
3) PyTorch
4) Matplotlib
5) Sklearn
6) Scikit-learn

# Tutorial
1) Download the code and data. In the folder "data_QSIS_Event" and "data_QSIS_Noise", put your .sac files in corresponding folders.
2) Run: "QSIS_dataprep_event.ipynb".
3) Run: "QSIS_dataprep_noise.ipynb".
4) Run: "QSIS_dataprep_merge.ipynb".
5) Run: "model_train_Torch.ipynb" to pre-train model with STanford EArthquake Dataset.
6) Run: "model_finetune_Torch.ipynb" to finetune model with QSIS Dataset.
7) Check the results in the "Result_Training" and "Result_Finetuing" folder.
8) Run: "convert_to_ONNX.py" to convert the PyTorch model into ONNX model.
 
The ANN architecture includes the following laters: Conv2D, MaxPool, DropOut, Conv2D, MaxPool, DropOut, Conv2D, MaxPool, DropOut, FC16(relu), FC3(softmax)
The code generates plots of loss function, validation curve and some examples of seismic waves and noise.
The overall accuracy for the pre-train model is 93%.

## Code of data preparation functions

```python
import numpy as np
import scipy.signal as signal

def quantize(A, dtype = np.int16):
    'quantize float data in range [-127,127]'
    m = np.max(np.abs(A), axis = (1, 2), keepdims = True)
    factors = np.iinfo(dtype).max / m 
    return (A * factors).astype(dtype = dtype)

def remove_small_amplitude(A, B, min_amp = 1e-8):
    to_keep = np.where(np.max(np.abs(A), axis=(1,2)) > min_amp)[0]
    return A[to_keep], B[to_keep]

def remove_large_amplitude(A, B, max_amp = 2.0 * 9.81):
    to_keep = np.where(np.max(np.abs(A), axis = (1,2)) < max_amp)[0]
    return A[to_keep], B[to_keep]

def detrend(X):
    'Remove mean and trend from data'
    N = X.shape[-1]
    # create linear trend matrix 
    A = np.zeros((2, N),dtype = X.dtype)
    A[1,:] = 1
    A[0,:] = np.linspace(0, 1, N)
    R = A @ np.transpose(A)
    Rinv = np.linalg.inv(R)
    factor = np.transpose(A) @ Rinv
    X -= (X @ factor) @ A
    return X

def taper(A, alpha):
    'taper signal'
    window = signal.tukey(A.shape[-1],alpha)
    A *= window
    return A

def shift_event(data, maxshift, rate, start, halfdim): 
    'Randomly rotate the array to shift the event location'
    
    if np.random.uniform(0, 1) < rate:
        start += int(np.random.uniform(-maxshift, maxshift))             
    return data[:, start-halfdim:start + halfdim]

def adjust_amplitude_for_multichannels(data):
    'Adjust the amplitude of multi-channel data'
    
    tmp = np.max(np.abs(data), axis = -1, keepdims = True)
    assert(tmp.shape[0] == data.shape[0])
    if np.count_nonzero(tmp) > 0:
        data *= data.shape[0] / np.count_nonzero(tmp)
    return data

def scale_amplitude(data, rate):
    'Scale amplitude or waveforms'
    
    tmp = np.random.uniform(0, 1)
    if tmp < rate:
        data *= np.random.uniform(1, 3)
    elif tmp < 2*rate:
        data /= np.random.uniform(1, 3)
    return data

def normalize(data):
    'Normalize waveforms over each event'
        
    max_data = np.max(data, axis=(1, 2), keepdims=True)
    assert(max_data.shape[0] == data.shape[0])
    max_data[max_data == 0] = 1
    data /= max_data              
    return data
```

## Code of model training function
```python
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    epoch_arr = np.array([])
    lr_arr = np.array([])
    train_loss_arr = np.array([])
    train_acc_arr = np.array([])

    val_loss_arr = np.array([])
    val_acc_arr = np.array([])

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # These lists will hold all labels and predictions for this epoch
            # all_labels = []
            # all_preds = []

            all_labels = np.array([])
            all_preds = np.array([])

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs
                labels = labels

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    _, labels = torch.max(labels.data, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Add current labels and predictions to the lists
                # all_labels.extend(labels.tolist())
                # all_preds.extend(preds.tolist())

                all_labels = np.append(all_labels, labels.tolist())
                all_preds = np.append(all_preds, preds.tolist())

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # After each epoch, compute and print the confusion matrix
            if phase == 'val':
                print("Confusion Matrix:")
                cm_here = confusion_matrix(all_labels, all_preds)
                print(cm_here)

            # save the loss and accurracy
            if phase == 'train':
                train_loss_arr = np.append(train_loss_arr, epoch_loss)
                train_acc_arr = np.append(train_acc_arr, epoch_acc.detach().cpu())
                lr_arr = np.append(lr_arr, optimizer.param_groups[0]['lr'])
                epoch_arr = np.append(epoch_arr, epoch)
            elif phase == 'val':
                val_loss_arr = np.append(val_loss_arr, epoch_loss)
                val_acc_arr = np.append(val_acc_arr, epoch_acc.detach().cpu())

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)

    model_class = MyModel(model = model, 
                          epoch_arr = epoch_arr, 
                          train_loss_arr = train_loss_arr, 
                          val_loss_arr = val_loss_arr, 
                          train_acc_arr = train_acc_arr, 
                          val_acc_arr = val_acc_arr, 
                          lr_arr = lr_arr, 
                          cm = cm_here, 
                          best_acc = best_acc)

    return model_class
```

## Time Series Model Definition

```python
class EarthquakeCNN(nn.Module):
    def __init__(self):
        super(EarthquakeCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = (3, 1), stride = (1, 1))
        self.bn1 = nn.BatchNorm2d(16)
        self.GELU1 = nn.GELU()
        self.maxpool1 = nn.MaxPool2d(kernel_size = (3, 1), stride = (3, 1))
        self.dropout1 = nn.Dropout(p = 0.1)

        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (3, 1), stride = (1, 1))
        self.bn2 = nn.BatchNorm2d(32)
        self.GELU2 = nn.GELU()
        self.maxpool2 = nn.MaxPool2d(kernel_size = (3, 1), stride = (3, 1))
        self.dropout2 = nn.Dropout(p = 0.1)

        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3, 1), stride = (1, 1))
        self.bn3 = nn.BatchNorm2d(64)
        self.GELU3 = nn.GELU()
        self.maxpool3 = nn.MaxPool2d(kernel_size = (3, 1), stride = (3, 1))
        self.dropout3 = nn.Dropout(p = 0.1)

        self.fc1 = nn.Linear(384, 64)
        self.GELU4 = nn.GELU()
        self.fc2 = nn.Linear(64, 16)
        self.GELU5 = nn.GELU()
        self.fc3 = nn.Linear(16, 3)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.GELU1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.GELU2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.GELU3(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.GELU4(x)
        x = self.fc2(x)
        x = self.GELU5(x)
        x = self.fc3(x)

        return x
```


## Spectrogram Model Definition

```python
class SpectrogramCNN(nn.Module):
    def __init__(self):
        super(SpectrogramCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 48, kernel_size = (3, 3), stride = (1, 1))
        self.bn1 = nn.BatchNorm2d(48)
        self.GELU1 = nn.GELU()
        self.maxpool1 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
        self.dropout1 = nn.Dropout(p = 0.1)

        self.conv2 = nn.Conv2d(in_channels = 48, out_channels = 96, kernel_size = (3, 3), stride = (1, 1))
        self.bn2 = nn.BatchNorm2d(96)
        self.GELU2 = nn.GELU()
        self.maxpool2 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
        self.dropout2 = nn.Dropout(p = 0.1)

        self.conv3 = nn.Conv2d(in_channels = 96, out_channels = 192, kernel_size = (3, 3), stride = (1, 1))
        self.bn3 = nn.BatchNorm2d(192)
        self.GELU3 = nn.GELU()
        self.maxpool3 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
        self.dropout3 = nn.Dropout(p = 0.1)

        self.fc1 = nn.Linear(32640, 1280)
        self.GELU4 = nn.GELU()
        self.fc2 = nn.Linear(1280, 64)
        self.GELU5 = nn.GELU()
        self.fc3 = nn.Linear(64, 2)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.GELU1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.GELU2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.GELU3(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.GELU4(x)
        x = self.fc2(x)
        x = self.GELU5(x)
        x = self.fc3(x)

        return x
```
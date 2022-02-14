import os
import shutil
import librosa
import numpy as np
import librosa.display
from pydub import AudioSegment
import matplotlib.pyplot as plt

# Original dataset path
original_path = './data/genres_original'

# Cropped 3s audios path
cropped_path = './data/genres_3s'

# Train data path
train_path = './data/train'

if(os.path.exists(cropped_path)):
    shutil.rmtree(cropped_path)
os.makedirs(cropped_path)

if(os.path.exists(train_path)):
    shutil.rmtree(train_path)
os.makedirs(train_path)

# Cropping 30s audios into 3s audios and extracting features
for genre in os.listdir(original_path):
    os.makedirs(f'{cropped_path}/{genre}')
    os.makedirs(f'{train_path}/{genre}')

    i = 0
    for original in os.listdir(f'{original_path}/{genre}'):
        for each in range(10):
            t1 = 3 * each * 1000
            t2 = 3 * (each + 1) * 1000

            try:
                # Cropping audios
                cropped = AudioSegment.from_wav(f'{original_path}/{genre}/{original}')
                cropped = cropped[t1:t2]

                number = "{:05d}".format(i)
                
                # Export new cropped audio
                new = f'{cropped_path}/{genre}/{genre}.{number}.wav'
                cropped.export(new, format='wav')

                y, sr = librosa.load(new, duration=3)
                                
                # Extracting spectogram
                spec = librosa.feature.melspectrogram(y=y, sr=sr)
                librosa.display.specshow(librosa.power_to_db(spec, ref=np.max))

                # Preparing spectogram image
                plt.gca().set_axis_off()
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                            hspace = 0, wspace = 0)
                plt.margins(0,0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())

                # Save image
                plt.savefig(f'{train_path}/{genre}/{genre}.{number}.png', pad_inches=0, bbox_inches='tight')

                i += 1
            except:
                continue

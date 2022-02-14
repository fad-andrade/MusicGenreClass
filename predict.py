import os
import sys
import json
import shutil
import librosa
import numpy as np
import librosa.display
import tensorflow as tf
from pydub import AudioSegment
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array

def MP3ToWAV(song, temp_path):
    audio = AudioSegment.from_mp3(song)
    
    new_audio_name = f'{temp_path}/song.wav'
    audio.export(new_audio_name, format="wav")

    return new_audio_name

def extract3s(song, temp_path):
    snippets = []
    audio = AudioSegment.from_wav(song)
    duration = int(AudioSegment.from_wav(song).duration_seconds)
    
    while(duration % 3 != 0):
        duration -= 1
    
    times = int(duration / 3)
    for each in range(times):
        t1 = 3 * each * 1000
        t2 = 3 * (each + 1) * 1000
        
        cropped = audio[t1 : t2]

        new_audio_name = f'{temp_path}/extracted_{each}.wav'
        cropped.export(new_audio_name, format="wav")
        snippets.append(new_audio_name)

    return snippets

def extractFeature(song,  temp_path):
    y, sr = librosa.load(song, duration=3)
                    
    # Extracting spectogram
    spec = librosa.feature.melspectrogram(y=y, sr=sr)
    librosa.display.specshow(librosa.power_to_db(spec, ref=np.max))

    # Preparing spectogram image
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    # Save image
    image_name = f'{temp_path}/spec.png' 
    plt.savefig(image_name, pad_inches=0, bbox_inches='tight')
    
    # Get image as array
    img = load_img(image_name, color_mode='rgba', target_size=(288, 432))
    
    data_img = img_to_array(img)
    data_img = np.reshape(data_img, (1, 288, 432, 4))
    
    return data_img

def main():		
	song_path = sys.argv[1]
	
	temp_path = './temp'
	if(os.path.exists(temp_path)):
		shutil.rmtree(temp_path)
	os.mkdir(temp_path)

	wav_song = MP3ToWAV(song_path, temp_path)

	snippets = extract3s(wav_song, temp_path)

	model = tf.keras.models.load_model('my_model.h5')

	labels = []
	for snip in snippets:
		data_img = extractFeature(snip, temp_path)
		
		pred = model.predict(data_img/255)
		pred = pred.reshape((10,))

		label = np.argmax(pred)
		labels.append(label)
	
	gender_count = []
	gender_percentage = {
							'Blues': None, 
							'Classical': None,
							'Country': None,
							'Disco': None,
							'HipHop': None,
							'Jazz': None,
							'Metal': None,
							'Pop': None,
							'Reggae': None,
							'Rock': None
						}
	
	for i in range(len(gender_percentage)):
		gender_count.append(labels.count(i))
	
	for gender, count in zip(gender_percentage, gender_count):
		gender_percentage[gender] = round((count * 100) / sum(gender_count), 2)
	gender_percentage = sorted(gender_percentage.items(), key=lambda x: x[1], reverse=True)
	
	print(song_path)
	for each in gender_percentage:
		if(each[1] > 0):
			print(f'{each[0]}: {each[1]}%')
	print()
	
	shutil.rmtree(temp_path)
	
main()

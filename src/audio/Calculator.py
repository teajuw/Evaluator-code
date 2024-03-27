from librosa import *
import matplotlib.pyplot as plt
import math
import re
import time

from music21 import *
import numpy as np
import pandas as pd

'''
    This class takes in an audio buffer as input and generates a dataframe
'''
class Calculator:
    
    
    '''
    Takes in a audio buffer and returns a pandas dataframe calculated using Librosa
    
    :param buffer: the audio buffer generated by AudioThread
    :param fast: if fast, will use the faster yin() function for calculations. 
                 if not, will use the more accurate pyin() function for calculations
    :param create_file: store the generated dataframe in a file at out_file
    :param out_file: path to store generated dataframe
    :param verbose: prints calculation times and the dataframe
    :param rms_graph: graphs the Root mean square
    :return: the pandas dataframe
    
    '''
    def calculate(self, buffer, fast=True, create_file=False, out_file="", verbose=True, rms_graph=False):
        start = time.time()
        #Librosa calculations
        numpy_array = np.frombuffer(buffer, dtype=np.float64)
        if fast:
            f0 = yin(y=buffer,
                fmin=note_to_hz('A0'),
                fmax=note_to_hz('C7'), sr=44100)
            
        else:
            f0, voiced_flag, voiced_probs = pyin(y=buffer,
                                                    fmin=note_to_hz('A0'),
                                                    fmax=note_to_hz('C7'), sr=44100)
        
        
        if verbose:
            end_librosa = time.time()
            print("Librosa took", end_librosa - start, "seconds")

        #replace NaN with 0s
        if (len(f0) > 0):
            f0 = np.nan_to_num(f0, nan=0, posinf=10000, neginf=-10000)
            
        #get the time each entry is recorded    
        times = times_like(f0, sr=44100)
                
        #The points in the array where a new note begins
        onset_frames = onset.onset_detect(y=buffer, sr=44100)
        
        #Notes based on onsets
        onset_freqs = []
        onset_notes = []
        onset_times = []
        for my_onset in onset_frames: 
            #not sure about the +3 here. But possibly onset is the note right before the switch
            #Adding 3 because frequency at onset isn't accurate
            #Definitely need something better
            if not my_onset + 3 >= len(f0):
                my_onset += 3
            else:
                my_onset = len(f0) - 1
            freq = f0[my_onset]
            time_pos = times[my_onset]
            onset_freqs.append(freq)
            onset_times.append(time_pos)
        
        notes = self._note_names_from_freqs(f0, 0)
        onset_notes = self._note_names_from_freqs(onset_freqs, 0)
        print(notes)
        print(onset_notes)
        
        #Calculate durations:
        
        durs = self._get_durations(onset_times, times[len(times) - 1])
        
        #Calculates RMS value of each entry
        
        if rms_graph:
            rms = feature.rms(y=buffer)
            _graph_rms(rms[0])
        
        my_dict = {'Note Name': onset_notes, 'Frequency': onset_freqs, 'Times': onset_times, 'Duration': durs}
        note_names = list(my_dict['Note Name'])
        note_names = [note.replace('♯', '#') for note in note_names] #Solves weird character issue
        my_dict['Note Name'] = note_names
        df = pd.DataFrame(data=my_dict)
        
        offs = []
        for i in range (len(df['Note Name'])):
            note_str = df['Note Name'][i]
            note = note_str
            offset = 0
            if (note_str != "rest"):
                print(note_str)
                note, offset = [x for x in re.split('([A-G][#-]?[0-9]+)([-+][0-9]+)', note_str) if x]
                if offset[0] == '+':
                    offset = int(offset[1:])
                else:
                    offset = int(offset)
                df.loc[i, 'Note Name'] = note
                offs.append(offset)
            else:
                offs.append(0) #offset (cents) of 0
            
        df.insert(4, "Cents", offs)
    
        if create_file:
            df.to_csv("out.csv")
            
        if verbose:
            end_calc = time.time()
            print(df)
            print("Calculation took", end_calc - start, "seconds")
        return df

    '''
    Calculate the Note corresponding to the frequency 
    Will set note name to Rest if frequency is below a certain frequency
    Will return an array of note names corresponding to each entry in given array
    '''
    def _note_names_from_freqs(self, f0: np.ndarray, rest_threshold:int=0):
        notes = []
        if len(f0) > 0:
            for freq in f0:
                if freq <= rest_threshold:
                    notes.append('rest')
                else:
                    notes.append(hz_to_note(freq, cents=True))
        return notes

    '''
    Given times at each index and time of ended recording, return duration of each note
    '''
    def _get_durations(self, onset_times, end_time):
        durs = []
        for i in range(len(onset_times)):
            if i < len(onset_times) - 1:
                durs.append(onset_times[i + 1] - onset_times[i])
            else:
                durs.append(end_time - onset_times[i])
        return durs

    '''
    Takes in rms array and graphs it
    '''
    def _graph_rms(self, rms, color="green"):
        #Graph rms
        x = np.arange(0, len(rms))
        plt.title("Line graph") 
        plt.xlabel("X axis") 
        plt.ylabel("Y axis") 
        plt.plot(x, rms, color =color) 
        plt.show()

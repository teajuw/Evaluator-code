import music21
from music21 import *
import numpy as np
import pandas as pd
from typing import *
import math

class PartExtractor:
    def __init__(self, score_path: str) -> None:
        self.score = converter.parse(score_path)
        self.solo_part = None
        self.accompaniment_part = None
    
    def extract_parts(self) -> None:
        parts = self.score.parts
        self.solo_part = parts[0]
        self.accompaniment_part = parts[1]
    
    def generate_dataframe_from_part(self, part: music21.stream.Part) -> pd.DataFrame:
        measures = len(part.getElementsByClass('Measure'))

        tempo_changes = {}
        for i in range(1, measures+1, 1):
            measure = part.measure(i)
            for j in range(0, len(measure)):
                if isinstance(measure[j], music21.tempo.MetronomeMark):
                    tempo_changes[measure.number] = measure[j].getQuarterBPM()
        if not tempo_changes:
            tempo_changes[1] = 120

        durations = []

        for i in range(1, measures+1, 1):
            if part.measure(i) is None:
                continue
            lengths = []

            if part.measure(i).hasVoices():
                measure = part.measure(i).voices[0].flatten().notesAndRests
            else: 
                measure = part.measure(i).flatten().notesAndRests
            
            for j in range(0, len(measure)):
                s = measure[j].duration.quarterLength
                lengths.append(float(s))
            durations.append(lengths)

        measure_notes = []
        measure_notes_frequency = []

        for i in range(1, measures+1, 1):
            if part.measure(i) is None:
                continue
            notes = []
            notes_frequency = []

            if part.measure(i).hasVoices():
                measure = part.measure(i).voices[0].flatten().notesAndRests
            else:
                measure = part.measure(i).flatten().notesAndRests
            
            for j in range(0, len(measure)):
                if (measure[j].isChord):
                    chord = measure[j].notes
                    notes.append(str(chord[-1].pitch.name + str(chord[-1].pitch.octave)))
                    notes_frequency.append(chord[-1].pitch.frequency)
                    continue
                elif (measure[j].isRest):
                    s = 'rest'
                    f = 0.0
                else:
                    f = measure[j].pitch.frequency
                    s = str(measure[j].pitch.name)
                    s += str(measure[j].pitch.octave)
                notes.append(s)
                notes_frequency.append(f)
            measure_notes.append(notes)
            measure_notes_frequency.append(notes_frequency)

        bpm = tempo_changes[1]
        quarter_note_duration = (1 / bpm) * 60
        note_duration = []
        for measure in durations:
            note_duration.append([note_length * quarter_note_duration for note_length in measure])
        new_durations = np.concatenate(durations)
        new_measure_notes = np.concatenate(measure_notes)
        new_measure_notes_frequency = np.concatenate(measure_notes_frequency)
        new_note_duration = np.concatenate(note_duration)

        start_times = []
        start_times.append(0)
        curr_time = 0

        for i in range(0, len(new_note_duration) - 1):
            curr_time += new_note_duration[i]
            start_times.append(curr_time)
        assert(len(new_measure_notes) == len(new_measure_notes_frequency))

        note_type = [duration.Duration(quarterLength=quarter_note_length).type for quarter_note_length in new_durations]
        df = pd.DataFrame({'Note Type': note_type, 'Duration': new_note_duration, 
                   'Note Name': new_measure_notes, 'Note Frequency': new_measure_notes_frequency, 
                   'Start Time': start_times})
        return df
        
    def generate_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.solo_part is None or self.accompaniment_part is None:
            self.extract_parts()
        
        return (self.generate_dataframe_from_part(self.solo_part),
                self.generate_dataframe_from_part(self.accompaniment_part))

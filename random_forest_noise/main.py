# %%
# Install pretty_midi if needed
!pip install -q pretty_midi

# %%
import mido
import os
import pretty_midi
from IPython.display import Audio

# %%
# Get all files in the target directory
all_midi_files = [f for f in os.listdir('/root/class_stuff/musicml/assign2/midi_files/') if f.endswith('.mid')]

# %%
print(all_midi_files[0])
print(len(all_midi_files))

# %%
def get_instruments(midi_file):
    """
    Extracts the instruments from a MIDI file.
    """
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    return [instrument.name for instrument in midi_data.instruments if instrument.is_drum is False]

dataroot = '/root/class_stuff/musicml/assign2/midi_files/'

all_instruments = []
num_instruments = []
only_plpr = []
for midi_file in all_midi_files:
    full_path = os.path.join(dataroot, midi_file)
    instruments = get_instruments(full_path)
    format_instrument_names = [inst.lower() for inst in instruments]
    all_instruments.extend(format_instrument_names)
    num_instruments.append(len(instruments))
    if 'piano right' in format_instrument_names and 'piano left' in format_instrument_names:
        only_plpr.append(midi_file)
        if len(format_instrument_names) != 2:
            print(f"File {midi_file} has more than two instruments: {format_instrument_names}") 

print(f'Tota number of MIDI files: {len(all_midi_files)}')
unique_instruments, counts = np.unique(all_instruments, return_counts = True)
for i, inst in enumerate(unique_instruments):
    print(f"{inst}: {counts[i]}")

print(np.mean(num_instruments))
print(len(only_plpr))



# %%
import pretty_midi
from IPython.display import Audio
from collections import defaultdict
import numpy as np

def play_midi(midi_file_path, print_info=True):
    """Play a MIDI file and print information including chord progression"""
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    
    # Print basic information
    if print_info:
        print(f"Processing MIDI file: {midi_file_path}")
        print(f"Duration: {midi_data.get_end_time():.2f} seconds")
        print(f"Number of instruments: {len(midi_data.instruments)}")
    
    # Extract chord progression
    # Group notes into time-based chunks for chord analysis
    time_resolution = 0.1
    max_time = midi_data.get_end_time()
    chord_sequence = []
    
    # Note name mapping
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Create bins for each time segment
    for t in np.arange(0, max_time, time_resolution):
        active_notes = []
        
        # Collect all notes active at time t
        for instrument in midi_data.instruments:
            if instrument.is_drum:
                continue
                
            for note in instrument.notes:
                if note.start <= t < note.end:
                    pitch = note.pitch % 12  # Get pitch class (0-11)
                    active_notes.append(pitch)
        
        # If we have notes at this time, identify the chord
        if active_notes:
            # Count occurrences of each pitch class
            pitch_classes = [0] * 12
            for note in active_notes:
                pitch_classes[note] += 1
                
            # Find the root note (most frequent or lowest)
            root = active_notes[0]
            if len(active_notes) > 1:
                max_count = 0
                for i, count in enumerate(pitch_classes):
                    if count > max_count:
                        max_count = count
                        root = i
            
            # Convert to chord name (simplified)
            chord_notes = sorted(list(set(active_notes)))
            chord_name = note_names[root]
            
            # Detect chord quality (very basic detection)
            if len(chord_notes) >= 3:
                # Check for major/minor
                has_major_third = (root + 4) % 12 in chord_notes
                has_minor_third = (root + 3) % 12 in chord_notes
                has_fifth = (root + 7) % 12 in chord_notes
                
                if has_major_third:
                    chord_name += " maj"
                elif has_minor_third:
                    chord_name += " min"
                
                # Check for seventh
                has_maj_seventh = (root + 11) % 12 in chord_notes
                has_min_seventh = (root + 10) % 12 in chord_notes
                
                if has_maj_seventh:
                    chord_name += "7"
                elif has_min_seventh and has_major_third:
                    chord_name += "7"
                elif has_min_seventh and has_minor_third:
                    chord_name += "7"
            
            chord_sequence.append(chord_name)
    
    # Print chord progression
    if print_info:
        print("\nChord Progression:")
        # Remove consecutive duplicates
        unique_progression = []
        for chord in chord_sequence:
            if not unique_progression or unique_progression[-1] != chord:
                unique_progression.append(chord)
        print(" → ".join(unique_progression[:20]))
        if len(unique_progression) > 20:
            print(f"... and {len(unique_progression) - 20} more chords")
    
    # Print instrument details
    for i, instrument in enumerate(midi_data.instruments):
        notes_count = len(instrument.notes)
        if notes_count > 0:
            print(f"Instrument {i}: {instrument.name}, Notes: {notes_count}")

    return midi_data, unique_progression

test_file = '/root/class_stuff/musicml/assign2/midi_files/' + only_plpr[0]

midi_data, unique_chords = play_midi(test_file, print_info=True)
audio_data = midi_data.synthesize(fs=44100)
Audio(audio_data, rate=44100, autoplay=True)

# %%
!pip install -q music21

# %%
from music21 import converter, chord, stream, tempo
import numpy as np
import pretty_midi
from IPython.display import Audio

# Load and chordify
score = converter.parse(test_file)
chords = score.chordify()

# Get tempo
metronome = score.recurse().getElementsByClass(tempo.MetronomeMark).first()
bpm = metronome.number if metronome else 120
print(f"Detected BPM: {bpm}")

# Get the total duration of the piece in quarter lengths
duration = chords.highestTime
print(f"Total duration (quarter lengths): {duration}")

# Sampling interval (1 = one beat)
interval = 1.0
time_steps = np.arange(0, duration, interval)

# Extract the "active" chord at each time step
chord_stream = stream.Part()
chord_stream.insert(0, tempo.MetronomeMark(number=bpm))

for t in time_steps:
    # Get all notes sounding at this time
    elements = chords.flat.getElementsByOffset(t, mustBeginInSpan=False)
    for e in elements:
        if isinstance(e, chord.Chord):
            c = chord.Chord(e)
            c.quarterLength = interval
            chord_stream.insert(t, c)
            break  # Only use the first chord found at this time

# Save MIDI
chord_stream.write('midi', fp='chord_progression_fixed_intervals.mid')

# Synthesize
midi_data = pretty_midi.PrettyMIDI('chord_progression_fixed_intervals.mid')
audio_data = midi_data.synthesize(fs=44100)
Audio(audio_data, rate=44100, autoplay=True)

# %%
midi_data = pretty_midi.PrettyMIDI(test_file)
audio_data = midi_data.synthesize(fs=44100)
Audio(audio_data, rate=44100, autoplay=True)

# %%
def play_midi(midi_file_path, print_info=True):
    """Play a MIDI file and extract chord progressions separately for piano right and left hands"""
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    
    # Print basic information
    if print_info:
        print(f"Processing MIDI file: {midi_file_path}")
        print(f"Duration: {midi_data.get_end_time():.2f} seconds")
        print(f"Number of instruments: {len(midi_data.instruments)}")
    
    # Identify piano right and piano left instruments
    piano_right_idx = None
    piano_left_idx = None
    
    for i, instrument in enumerate(midi_data.instruments):
        if instrument.name.lower() == 'piano right':
            piano_right_idx = i
        elif instrument.name.lower() == 'piano left':
            piano_left_idx = i
    
    # Get the instrument programs from original
    right_program = midi_data.instruments[piano_right_idx].program
    left_program = midi_data.instruments[piano_left_idx].program
    
    # Extract chord progressions separately for each hand
    time_resolution = 0.1
    max_time = midi_data.get_end_time()
    
    # Note name mapping
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    right_chord_sequence = []
    left_chord_sequence = []
    
    # Store original note data for reference
    right_note_data = []
    left_note_data = []
    
    # Process right hand and store active notes for each timepoint
    for t in np.arange(0, max_time, time_resolution):
        active_notes = []
        note_details = []  # Store detailed note information
        
        # Collect all notes active at time t for piano right
        instrument = midi_data.instruments[piano_right_idx]
        for note in instrument.notes:
            if note.start <= t < note.end:
                pitch = note.pitch % 12  # Get pitch class (0-11)
                active_notes.append(pitch)
                # Store full note info
                note_details.append({
                    'pitch': note.pitch,
                    'velocity': note.velocity
                })
        
        # If we have notes at this time, identify the chord
        if active_notes:
            chord_name = extract_chord_name(active_notes, note_names)
            right_chord_sequence.append(chord_name)
            right_note_data.append(note_details)
        else:
            right_chord_sequence.append(None)  # No notes at this time
            right_note_data.append([])
    
    # Process left hand and store active notes for each timepoint
    for t in np.arange(0, max_time, time_resolution):
        active_notes = []
        note_details = []  # Store detailed note information
        
        # Collect all notes active at time t for piano left
        instrument = midi_data.instruments[piano_left_idx]
        for note in instrument.notes:
            if note.start <= t < note.end:
                pitch = note.pitch % 12  # Get pitch class (0-11)
                active_notes.append(pitch)
                # Store full note info
                note_details.append({
                    'pitch': note.pitch,
                    'velocity': note.velocity
                })
        
        # If we have notes at this time, identify the chord
        if active_notes:
            chord_name = extract_chord_name(active_notes, note_names)
            left_chord_sequence.append(chord_name)
            left_note_data.append(note_details)
        else:
            left_chord_sequence.append(None)  # No notes at this time
            left_note_data.append([])
    
    # Find segments with unique chords
    right_unique = []
    right_timings = []
    right_details = []
    current_right = None
    
    for i, chord in enumerate(right_chord_sequence):
        if chord != current_right:
            current_right = chord
            if chord is not None:
                right_unique.append(chord)
                right_timings.append(i * time_resolution)
                right_details.append(right_note_data[i])
    
    left_unique = []
    left_timings = []
    left_details = []
    current_left = None
    
    for i, chord in enumerate(left_chord_sequence):
        if chord != current_left:
            current_left = chord
            if chord is not None:
                left_unique.append(chord)
                left_timings.append(i * time_resolution)
                left_details.append(left_note_data[i])
    
    # Print chord progressions
    if print_info:
        print("\nRight Hand Chord Progression:")
        print(" → ".join([c for c in right_unique[:20] if c is not None]))
        if len(right_unique) > 20:
            print(f"... and {len(right_unique) - 20} more chords")
        
        print("\nLeft Hand Chord Progression:")
        print(" → ".join([c for c in left_unique[:20] if c is not None]))
        if len(left_unique) > 20:
            print(f"... and {len(left_unique) - 20} more chords")
    
    # Create a new MIDI file with both hands
    chord_midi = pretty_midi.PrettyMIDI()
    
    # Create instruments for right and left hands with original programs
    right_piano = pretty_midi.Instrument(program=right_program, name='piano right')
    left_piano = pretty_midi.Instrument(program=left_program, name='piano left')
    
    # Create a mapping from note names to numbers
    note_to_number = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 
                     'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
    
    # Add notes for right hand chord progression
    for i in range(len(right_unique)):
        chord_name = right_unique[i]
        start_time = right_timings[i]
        end_time = right_timings[i+1] if i+1 < len(right_timings) else max_time
        note_details = right_details[i]
        
        # Use original voicing details when available
        enhanced_add_chord(chord_name, start_time, end_time, right_piano, note_to_number, note_details)
    
    # Add notes for left hand chord progression
    for i in range(len(left_unique)):
        chord_name = left_unique[i]
        start_time = left_timings[i]
        end_time = left_timings[i+1] if i+1 < len(left_timings) else max_time
        note_details = left_details[i]
        
        # Use original voicing details when available
        enhanced_add_chord(chord_name, start_time, end_time, left_piano, note_to_number, note_details)
    
    # Add the instruments to the MIDI file
    chord_midi.instruments.append(right_piano)
    chord_midi.instruments.append(left_piano)
    
    return midi_data, (right_unique, left_unique), chord_midi

def enhanced_add_chord(chord_name, start_time, end_time, instrument, note_to_number, note_details):
    """Add chord notes to an instrument using details from original MIDI when available"""
    if chord_name is None:
        return
    
    # If we have original note details, prefer using those directly
    if note_details and len(note_details) >= 3:  # If we have enough notes for a chord
        for note_info in note_details:
            note = pretty_midi.Note(
                velocity=note_info['velocity'],
                pitch=note_info['pitch'],
                start=start_time,
                end=end_time
            )
            instrument.notes.append(note)
        return
    
    # Otherwise, parse chord name and generate notes
    if chord_name.startswith(tuple(note_to_number.keys())):
        if len(chord_name) > 1 and chord_name[1] == '#':
            root = chord_name[:2]
            quality = chord_name[2:].strip()
        else:
            root = chord_name[0]
            quality = chord_name[1:].strip()
    else:
        print(f"Warning: Could not parse chord {chord_name}")
        return
    
    # Get root note number
    root_num = note_to_number[root]
    
    # Determine base octave from instrument name or existing notes
    avg_pitch = 60  # Default middle C
    if note_details:
        # Use average pitch from original notes as reference
        pitches = [note['pitch'] for note in note_details]
        avg_pitch = sum(pitches) // len(pitches)
    else:
        # Default octaves based on hand
        avg_pitch = 72 if instrument.name == 'piano right' else 48
    
    # Root note octave
    base_octave = (avg_pitch // 12) * 12
    
    # Set up notes in the chord based on quality
    chord_notes = [root_num]  # Start with root note
    
    # Determine chord quality
    if "min" in quality:
        chord_notes.extend([root_num + 3, root_num + 7])
    elif quality == "" or "maj" in quality:
        chord_notes.extend([root_num + 4, root_num + 7])
    
    if "7" in quality:
        if "maj7" in quality:
            chord_notes.append(root_num + 11)
        else:
            chord_notes.append(root_num + 10)
    
    # Get average velocity from original notes
    velocity = 80  # Default
    if note_details:
        velocities = [note['velocity'] for note in note_details]
        if velocities:
            velocity = sum(velocities) // len(velocities)
    
    # Create and add notes
    for note_num in chord_notes:
        # Place near original octave
        midi_note = base_octave + (note_num % 12)
        
        # Ensure the note is in a reasonable range
        while midi_note > 108:  # Higher than C8
            midi_note -= 12
        while midi_note < 21:   # Lower than A0
            midi_note += 12
            
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=midi_note,
            start=start_time,
            end=end_time
        )
        instrument.notes.append(note)

# %%
midi_data, unique_chords, chord_midi = play_midi(test_file, print_info=True)
audio_data = midi_data.synthesize(fs=44100)
audio_data = chord_midi.synthesize(fs=44100)
Audio(audio_data, rate=44100, autoplay=True)

# %%
audio_data = midi_data.synthesize(fs=44100)
Audio(audio_data, rate=44100, autoplay=True)

# %%
print("Chord progression:", unique_chords[:10])

# %%
def chord_progression_to_midi(chords, output_file='chord_progression.mid', 
                             duration=0.5, octave=4):
    """
    Convert a chord progression to MIDI and play it.
    
    Parameters:
    chords: List of chord names
    output_file: Path to save the MIDI file
    duration: Duration of each chord in seconds
    octave: Base octave for the chords
    """
    # Define note name to number mapping
    note_to_number = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 
                      'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 
                      'A#': 10, 'B': 11}
    
    # Convert chord names to MIDI note lists
    chord_array = []
    base_pitch = octave * 12  # C4 = 60
    
    for chord_name in chords:
        # Parse chord name more robustly
        # Check if chord is like "E7" with no space
        root_name = chord_name
        quality = "maj"  # Default quality
        
        # Extract root and quality for formats like "E7", "Cmaj7", "Bmin", etc.
        for i, char in enumerate(chord_name):
            if i > 0 and (char.isdigit() or char.lower() == 'm'):
                root_name = chord_name[:i]
                quality = chord_name[i:]
                break
        
        # Also check for space-separated format like "E maj7"
        if " " in chord_name:
            parts = chord_name.split()
            root_name = parts[0]
            quality = parts[1] if len(parts) > 1 else "maj"
        
        # Get root note number
        if root_name not in note_to_number:
            print(f"Warning: Unknown note '{root_name}' in chord '{chord_name}'. Skipping.")
            continue
            
        root = note_to_number[root_name]
        
        # Build chord based on quality
        chord = [root + base_pitch]  # Start with root note
        
        if "m" in quality.lower() or "min" in quality.lower():
            chord.append(root + 3 + base_pitch)  # Minor 3rd
        else:
            chord.append(root + 4 + base_pitch)  # Major 3rd
            
        chord.append(root + 7 + base_pitch)  # Perfect 5th
        
        if "7" in quality:
            if "maj7" in quality:
                chord.append(root + 11 + base_pitch)  # Major 7th
            else:
                chord.append(root + 10 + base_pitch)  # Minor 7th
        
        chord_array.append(chord)
    
    # Create MIDI from chord array
    midi_obj = create_midi_from_chords(chord_array, output_file, duration=duration)
    
    # Play the chord progression
    audio_data = midi_obj.synthesize()
    print(f"Playing chord progression with {len(chords)} chords")
    return Audio(audio_data, rate=44100, autoplay=True)

# Example usage with your progression
chord_progression_to_midi(unique_chords, 'extracted_progression.mid')

# %%
def extract_and_play_chord_progression(midi_file_path, output_file='chord_progression.mid'):
    """Extract chord progression from a MIDI file, convert to a new MIDI, and play it"""
    # Load the MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    
    # Extract chord progression
    time_resolution = 0.25  # 250ms chunks
    max_time = midi_data.get_end_time()
    chords = []
    chord_times = []
    
    # Create bins for each time segment
    for t in np.arange(0, max_time, time_resolution):
        active_notes = []
        
        # Collect all notes active at time t
        for instrument in midi_data.instruments:
            if instrument.is_drum:
                continue
                
            for note in instrument.notes:
                if note.start <= t < note.end:
                    pitch = note.pitch
                    active_notes.append(pitch)
        
        # If we have notes at this time, add the chord
        if active_notes:
            # Remove duplicate pitches
            chord = sorted(list(set(active_notes)))
            chords.append(chord)
            chord_times.append(t)
    
    # Create a new MIDI file with just the chord progression
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)  # Piano
    
    # Add each chord as a block
    for i, (chord, start_time) in enumerate(zip(chords, chord_times)):
        # Determine where this chord ends (either at next chord or after time_resolution)
        if i < len(chord_times) - 1:
            end_time = chord_times[i+1]
        else:
            end_time = start_time + time_resolution
        
        # Create notes for each pitch in the chord
        for pitch in chord:
            note = pretty_midi.Note(
                velocity=80,
                pitch=pitch,
                start=start_time,
                end=end_time
            )
            piano.notes.append(note)
    
    # Add instrument to the MIDI
    pm.instruments.append(piano)
    
    # Write the MIDI file
    pm.write(output_file)
    
    # Print summary
    print(f"Extracted {len(chords)} chords from {midi_file_path}")
    print(f"Chord progression saved to {output_file}")
    
    # Play the chord progression
    audio_data = pm.synthesize(fs=44100)
    return Audio(audio_data, rate=44100, autoplay=True)

# Example usage
test_file = '/root/class_stuff/musicml/assign2/midi_files/' + all_midi_files[0]
extract_and_play_chord_progression(test_file)

# %%
import mido
import numpy as np

def midi_to_chords(midi_file, resolution=0.25):
    """
    Extract chords from a MIDI file by chunking notes into fixed time intervals.
    
    Parameters:
    midi_file: Path to the MIDI file
    resolution: Time resolution in seconds for chunking notes into chords
    
    Returns:
    List of lists where each inner list contains MIDI note numbers for a chord
    """
    mid = mido.MidiFile(midi_file)
    
    # Create a dictionary to hold notes by time chunk
    notes_by_chunk = {}
    current_time = 0
    ticks_per_beat = mid.ticks_per_beat
    
    # Calculate seconds per tick (approximate)
    # Assuming tempo of 120 BPM if not specified
    tempo = 500000  # default tempo (microseconds per beat)
    seconds_per_tick = tempo / (ticks_per_beat * 1000000)
    
    for track in mid.tracks:
        track_time = 0
        for msg in track:
            track_time += msg.time
            
            # Update tempo if tempo message is encountered
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                seconds_per_tick = tempo / (ticks_per_beat * 1000000)
            
            # Convert ticks to seconds
            seconds = track_time * seconds_per_tick
            
            # Determine which chunk this note belongs to
            chunk_index = int(seconds / resolution)
            
            # Store note information
            if msg.type == 'note_on' and msg.velocity > 0:
                if chunk_index not in notes_by_chunk:
                    notes_by_chunk[chunk_index] = []
                notes_by_chunk[chunk_index].append(msg.note)
    
    # Convert dictionary to sorted list of chords
    max_chunk = max(notes_by_chunk.keys()) if notes_by_chunk else 0
    chords = []
    
    for i in range(max_chunk + 1):
        if i in notes_by_chunk:
            # Remove duplicates within each chord
            chords.append(list(set(notes_by_chunk[i])))
        else:
            # Empty chord for silent segments
            chords.append([])
    
    # Filter out empty chords if desired
    chords = [chord for chord in chords if chord]
    
    return chords

# Example usage
chords = midi_to_chords('/root/class_stuff/musicml/assign2/midi_files/' + all_midi_files[0])
print(f"Number of chords extracted: {len(chords)}")
print(f"First 10 chords: {chords[:10]}")

# You can then use these chords with your create_midi_from_chords function

# %%
import pretty_midi
import numpy as np

def create_midi_from_chords(chord_array, output_file='output.mid', 
                           duration=0.5, velocity=100, instrument_name='Acoustic Grand Piano'):
    """
    Create a MIDI file from an array of chords.
    
    Parameters:
    chord_array: List of lists where each inner list contains MIDI note numbers for a chord
    output_file: Path to save the MIDI file
    duration: Duration of each chord in seconds
    velocity: Velocity (loudness) of each note (0-127)
    instrument_name: Name of the instrument to use
    
    Returns:
    PrettyMIDI object
    """
    # Create a PrettyMIDI object
    pm = pretty_midi.PrettyMIDI()
    
    # Create an Instrument instance
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(instrument_name)
    )
    
    # Iterate through the chord array
    for i, chord in enumerate(chord_array):
        # Start time for this chord
        start = i * duration
        
        # Add each note in the chord
        for note_number in chord:
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=note_number,
                start=start,
                end=start + duration
            )
            instrument.notes.append(note)
    
    # Add the instrument to the PrettyMIDI object
    pm.instruments.append(instrument)
    
    # Write out the MIDI file
    pm.write(output_file)
    
    return pm

# Example usage
# C major, F major, G major, C major chord progression
chord_progression = [
    [60, 64, 67],  # C major (C, E, G)
    [65, 69, 72],  # F major (F, A, C)
    [67, 71, 74],  # G major (G, B, D)
    [60, 64, 67]   # C major (C, E, G)
]

# Create the MIDI file
midi_obj = create_midi_from_chords(chord_progression, 'chord_progression.mid')

# Play the MIDI file directly in the notebook
from IPython.display import Audio
audio_data = midi_obj.synthesize()
Audio(audio_data, rate=44100)

# %%




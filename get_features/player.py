import pyaudio
import wave

wav_dir = "../data/today/wav/"
chunk = 1024
seconds = 5

def play_segment(start_time, end_time, programme):
    duration = end_time - start_time
    start_time, duration = start_time / 1000, duration / 1000
    # open wave file
    wav = wave.open(wav_dir + programme, 'rb')
    py_audio = pyaudio.PyAudio()
    stream = py_audio.open(format=py_audio.get_format_from_width(wav.getsampwidth()),
                       channels=wav.getnchannels(),
                       rate=wav.getframerate(),
                       output=True)
    # skip unwanted frames
    n_frames = int(start_time * wav.getframerate())
    wav.setpos(n_frames)
    # write desired frames to audio buffer
    n_frames = int(duration * wav.getframerate())
    frames = wav.readframes(n_frames)
    stream.write(frames)
    # close and terminate
    wav.close()
    stream.close()
    py_audio.terminate()

if __name__ == "__main__":
    import pickle
    topic_timings = pickle.load(open("topic_timings", "rb"))
    programme_names = topic_timings.keys()
    print "Which programme?"
    for index, programme in enumerate(programme_names):
        print str(index + 1) + ")\t" + programme
    chosen_programme = programme_names[int(raw_input(">> ")) - 1]
    programme_segments = topic_timings[chosen_programme]
    print "Which segment?"
    for index, time in enumerate(programme_segments):
        print str(index + 1) + ")\t" + str(time)
    segment_choice = int(raw_input(">> ")) - 1
    chosen_segment_start = topic_timings[chosen_programme][segment_choice]
    chosen_segment_end = topic_timings[chosen_programme][segment_choice + 1]
    print "Playing chosen segment..."
    play_segment(chosen_segment_start, chosen_segment_end, chosen_programme)
    print "Done.\n"

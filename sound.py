from gtts import gTTS
import os
import time

m = 'text that will be played'
language = 'bn'
speech = gTTS(m, lang=language, slow=False)
speech.save("speech.mp3")
os.system("start speech.mp3")
time.sleep(5)
os.remove("speech.mp3")
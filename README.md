# Audio Video Sync
# Description
You are an indie game developer who's thrown themselves on truly understanding how a ball bounces - but you have one small problem - you have no balls to bounce, and going to a nearby basketball court seems like an unnecessary hassle. Luckily, you've got your hands on some animations of circles bouncing around a screen, accompanied by audio files of their bounces. If you could align the video/audio files you just might be able to understand the bouncing mechanics and save yourself from stepping outside!

The animations depict multiple circles, each varying in size, color, and speed, bouncing around the screen. With each bounce, a sound effect is generated. Between the bounces, the circles move linearly at a consistent speed. However, imagining these animations as taking place on a basketball court, you anticipate encountering pebbles and rough patches that might cause the circles to deviate in direction and speed upon impact.

Additionally, there is exactly one rectangular obstacle somewhere on the screen, off which the circles can bounce. You can assume that no circles will be inside this obstacle.

Circles can be different sizes, different colours and different speeds. You can assume they are not overlapping in the first frame, but they can freely move across one another during the video.

The audio file consists of a 1D array of numbers. The numbers represent the amount of noise being produced by the bounces continually, hidden in surrounding noise. Because the balls and impact points vary, the bounce sounds vary in intensity.

The technical specifications of your materials include a video frame rate of 30 FPS and an audio sampling frequency of 44100Hz. Given these resources, your tasks are as follows:

A) Return the center coordinates (row, column) of the left-most circle on the screen. The left-most circle is the one you would encounter first when coming from the left. You can assume that there will be only 1 left-most circle. Note: notice that you should output the pixel coordinates in row-major order, i.e. the coordinate system starts in the top-left corner, with the x-axis running downwards.

B) Return the center coordinates (row, column) of the rectangle obstacle.

C) Count the number of distinguishable bounces in the audio file (if two bounces perfectly overlap and cannot be distinguished count them as one bounce).

D) Count the number of bounces in the video stream. Assume that no shapes will be close to the edges near the start/end of the video.

E) Unfortunately, when you started your experiments, the video began playing before the audio, leading to mismatched start times. As a result, the audio and video files are not synchronized, with the video running ahead of the audio. Your task is to figure out where in the video the audio actually starts, and align them by printing the video frame number that corresponds to the start of the audio file.

Input format:
inputVideoPath inputAudioPath (read from the standard input)

Output format:
Output answers to subtasks in 5 separate lines for each of the subtasks in order. If you don't have the answer to some answer leave a blank line.

A) two integers separated by a space representing left-most circle coordinates
B) two integers separated by a space representing rectangle center coordinates
C) a single integer representing number of distinguishable bounces in audio
D) a single integer representing number of bounces in video
E) a single integer representing the video frame number when the audio starts
Example:
Input:
path/to/0/video.mp4 path/to/0/audio.npy
Output:
47 55
146 179
5
5
8
Available packages
numpy
imageio
all packages from the standard python library
Additionally, feel free to use the simplified implementations of functions from scipy.signal available in the additional materials in the bottom of this task (scipy_filter.zip).

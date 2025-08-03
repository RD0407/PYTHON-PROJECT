# YouTube Video Downloader

A simple Python script to download YouTube videos by providing the video URL as a command-line argument.  
Built using the [pytube](https://github.com/pytube/pytube) library.

## Features

- Download YouTube videos in the highest available resolution.
- Prints basic video information (title and view count).
- Easy to use from the terminal.

## Usage

1. **Install Dependencies:**

   Make sure you’re in a virtual environment (recommended):

   ```bash
   python3 -m venv myenv
   source myenv/bin/activate
   pip install pytube

2. **Save the Script:**
Save your script as videoDownloader.py.

3. **Run the Script:**

python videoDownloader.py <YouTube-Video-URL>

## Example:

    python videoDownloader.py https://www.youtube.com/watch?v=dQw4w9WgXcQ

## Example Output

Title:  Rick Astley - Never Gonna Give You Up (Official Music Video)
Views:  1400000000

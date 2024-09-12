import yt_dlp

url = "https://www.youtube.com/watch?v=ssiZXK7mvJM"
output_file = 'youtube_live_stream.mp4'

def download_youtube_live_stream(url, output_path="output.mp4"):
    # Options for downloading the live stream with reduced size
    ydl_opts = {
        'format': 'bestvideo[height<=640]+bestaudio/best[height<=640]',  # Limit the resolution to 480p
        'outtmpl': output_path,  # Save the video with the specified name
        'live_from_start': True,  # Start downloading from the beginning of the live stream
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',  # Convert to mp4 format
        }],
        'postprocessor_args': [
            '-vf', 'scale=iw/2:ih/2',  # Optional: resize the video to half the original dimensions
            '-b:v', '500k',  # Set video bitrate to 500k (adjust as needed)
            '-b:a', '128k',  # Set audio bitrate to 128k
        ],
    }

    # Download the live stream
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

# Call the function to download and reduce the video size
download_youtube_live_stream(url, output_file)

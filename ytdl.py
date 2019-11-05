import youtube_dl
import os
import os.path

# youtube_video_url: str = "https://www.youtube.com/watch?v=zsEcLVHnxUM"
youtube_video_url: str = 'https://www.youtube.com/watch?v=45eVbLRNxbU'


def get_yt_video(url, save_dir="/Users/loki/Movies", file_name='video.mp4', make_dirs=True):

    if make_dirs and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_path = os.path.join(save_dir, file_name)

    # remove existing file
    if os.path.exists(file_path):
        os.remove(file_path)

    options = {
               'verbose': True,
               'simulate': False,
               'format': None,
               'restrictfilenames': True,
               'writedescription': False,
               'writeinfojson': False,
               'writeannotations': False,
               'writethumbnail': False,
               'write_all_thumbnails': False,
               'writesubtitles': False,
               'writeautomaticsub': False,
               'allsubtitles': False,
               'cachedir': save_dir,
               'noplaylist': True,
               'include_ads': False,
               'outtmpl': file_path
               }

    with youtube_dl.YoutubeDL(options) as ydl:
        ydl.download([url])

    return file_path


if __name__ == '__main__':
    youtube_video_url: str = 'https://www.youtube.com/watch?v=tbgtGQIygZQ'
    ret = get_yt_video(youtube_video_url, "/Users/loki/Movies")

    print(ret)

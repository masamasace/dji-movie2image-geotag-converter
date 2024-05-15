import Video2Image

frame_interval = 30
start_frame = 0

# when end_frame is set as 0, the script will run until the end of the sequential frames.
end_frame = 0
initial_parameters = (frame_interval, start_frame, end_frame)

csv_path = r""
movie_dir = r"/Users/ms/Library/CloudStorage/OneDrive-nagaokaut.ac.jp/01_Research/01_Field Survey/06_202401_Noto/08_2024-05-14_第8回調査/01_survey_data/01_camera/02_UAV/01_ukai/01_movie"

# 0: SRT file, 1: CSV file processed with AIRDATA.com
reference_gnss_data = 0
csv_encoding = "shift-jis"

Video2Image.generate_frames_with_geotag(initial_parameters, csv_path, movie_dir, reference_gnss_data)

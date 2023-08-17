import Video2Image

frame_interval = 20
start_frame = 0

# when end_frame is set as 0, the script will run until the end of the sequential frames.
end_frame = 0
initial_parameters = (frame_interval, start_frame, end_frame)

csv_path = "PATH-TO-CSV-FILE"
movie_dir = "PATH-TO-MOVIE-DIRECTORY"

# 0: SRT file, 1: CSV file processed with AIRDATA.com
reference_gnss_data = 0
csv_encoding = "shift-jis"

Video2Image.generate_frames_with_geotag(initial_parameters, csv_path, movie_dir, reference_gnss_data)

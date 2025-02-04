import Video2Image

### Constants (定数) ###
REF_SRT = 0
REF_CSV = 1

### Parameters (パラメータ) ###
frame_interval = 30     # 何フレームごとに画像を保存するか
start_frame = 0         # 何フレーム目から画像を保存するか
end_frame = 0           # 何フレーム目まで画像を保存するか（0の場合は最後まで）

csv_path = r"PATH_TO_CSV_FILE"      # CSVファイルのパス
movie_dir = r"PATH_TO_MOVIE_DIR"    # 動画ファイルのディレクトリパス

reference_gnss_data = REF_SRT       # GPSデータの参照元（SRTファイル or CSVファイル）
csv_encoding = "shift-jis"          # CSVファイルの文字コード

initial_parameters = (frame_interval, start_frame, end_frame)

Video2Image.generate_frames_with_geotag(initial_parameters, csv_path, movie_dir, reference_gnss_data)

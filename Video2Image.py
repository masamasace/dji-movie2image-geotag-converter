

from pathlib import Path
import pandas as pd
import pandas.tseries.offsets as offsets
import cv2
import datetime
import ffmpeg
import piexif
import numpy as np
import re


def _deg2rational(temp):
    degree = int(temp)
    minute = int((temp - degree) * 60)
    second = int(((temp - degree) * 60 - minute) * 60 * 10000)
    res = [(degree, 1), (minute, 1), (second, 10000)]
    return res


def _extract_srt_data(srt_path):
    
    print("SRT File: ", srt_path.name)
    attributes_values = []
    
    # Extracts frame attributes from the SRT content
    with open(srt_path, "r", encoding="utf-8") as file:
        while True:

            line_data_raw = []
            temp = file.readline()

            if temp:
                line_data_raw.append(temp)
            else:
                break
            
            for i in range(4):
                line_data_raw.append(file.readline())
            
            file.readline()

            frame_num = int(line_data_raw[0])
            temp_format = "%Y-%m-%d %H:%M:%S.%f"
            temp_time_str = line_data_raw[3][:-1]
            if temp_time_str.count(",") == 2:
                temp_time_str = temp_time_str.split(",")[0] + "." + temp_time_str.split(",")[1] + temp_time_str.split(",")[2]
            elif temp_time_str.count(",") == 1:
                temp_time_str = temp_time_str.split(",")[0] + "." + temp_time_str.split(",")[1] + "000"
            
            cur_time = datetime.datetime.strptime(temp_time_str, temp_format)

            temp_attributes = re.findall("[0-9a-zA-Z_.\/]+[\s]?:[\s]?[\-]?[0-9a-zA-Z.\/]+", line_data_raw[4])
            temp_attributes_values = [frame_num, cur_time] + [item.split(':')[-1].strip() for item in temp_attributes]

            attributes_values.append(temp_attributes_values)
            
            

    # fnum: f-value, ev: Exposure Value, ct: Color Temperature, focal_len: Focal Length, 
    # Still unknown: dzoom_ratio, delta
    col_name = ['frame_num', 'time'] + [item.split(':')[0].strip() for item in temp_attributes]
    data = pd.DataFrame(attributes_values, columns=col_name)
        
    return data
            
            
def generate_frames_with_geotag(initial_parameters, csv_path, movie_dir, reference_gnss_data, csv_encoding="utf-8"):
    
    # decompose parameters
    frame_interval, start_frame, end_frame_temp = initial_parameters

    # convert path strings to Path Instance
    csv_path = Path(csv_path)
    movie_dir = Path(movie_dir)
    movie_path = [temp for temp in movie_dir.glob("*.MP4")]

    # in case of SRT data
    if reference_gnss_data == 0:
        
        # path of SRT file
        for movie_path_each in movie_path:

            # make result directory
            result_dir_path = movie_path_each.parent / "res"
            result_dir_path.mkdir(exist_ok=True)

            # format srt_data to pandas dataframe
            srt_path_each = movie_path_each.parent / (movie_path_each.stem + ".SRT")
            srt_data = _extract_srt_data(srt_path_each)
            
            alt_label = "abs_alt" if "abs_alt" in srt_data.columns.values else "altitude"

            # get basic information of the movie file
            video_info = ffmpeg.probe(movie_path_each)
            nb_frames = int(video_info["streams"][0]["nb_frames"])
            if end_frame_temp == 0:
                end_frame = nb_frames
            else:
                end_frame = end_frame_temp
                
            
            
            # import movie
            movie_cap = cv2.VideoCapture(str(movie_path_each))
            
            for index_export_frame in range(start_frame, end_frame-1 , frame_interval):

                # create image from frame
                movie_cap.set(cv2.CAP_PROP_POS_FRAMES, index_export_frame)

                ret, frame = movie_cap.read()
                
                # read geo-corrdinations
                temp_latitude = srt_data.iloc[index_export_frame]["latitude"]
                temp_longitude = srt_data.iloc[index_export_frame]["longitude"]
                temp_absolute_altitude = srt_data.iloc[index_export_frame][alt_label]
                
                # check whether both latitude and longitude are correctly captured
                flag_lat_lon_captured = (temp_latitude == 0) and (temp_longitude == 0)
                
                if ret:
                    frame_name = movie_path_each.stem + "_" + "{:05}".format(index_export_frame) + ".jpeg"
                    image_path = result_dir_path / frame_name

                    # create clipped image
                    cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 100]) 

                    # create a new GPS tag with latitude, longitude, and altitude values
                    # please note that sign of lat or lon are not considered yet
                    gps_data = {
                        piexif.GPSIFD.GPSLatitudeRef: 'N',
                        piexif.GPSIFD.GPSLatitude: _deg2rational(float(temp_latitude)),
                        piexif.GPSIFD.GPSLongitudeRef: 'E',
                        piexif.GPSIFD.GPSLongitude: _deg2rational(float(temp_longitude)),
                        piexif.GPSIFD.GPSAltitudeRef: 0,
                        piexif.GPSIFD.GPSAltitude: (int(float(temp_absolute_altitude) * 1000), 1000)
                    }

                    shutter_speed_numerator = int(srt_data.iloc[index_export_frame]["shutter"].split("/")[0])
                    shutter_speed_denominator = int(float(srt_data.iloc[index_export_frame]["shutter"].split("/")[1]))
                    shutter_speed = (shutter_speed_numerator, shutter_speed_denominator)

                    # something wrong with value's formatting.
                    # only DateTimeOriginal, FNumber and FocalLengthIn35mmFilm are correctly stored into generated Exif. Others are not.
                    exif_data = {
                        piexif.ExifIFD.DateTimeOriginal: srt_data.iloc[index_export_frame]["time"].strftime("%Y:%m:%d %H:%M:%S"),
                        piexif.ExifIFD.ISOSpeed: int(srt_data.iloc[index_export_frame]["iso"]), # Long
                        piexif.ExifIFD.ShutterSpeedValue: shutter_speed, # SRational
                        piexif.ExifIFD.FNumber: (int(srt_data.iloc[index_export_frame]["fnum"]), 100), ## Rational
                        piexif.ExifIFD.ExposureTime: (int(srt_data.iloc[index_export_frame]["ev"]), 100), # Rational
                        piexif.ExifIFD.FocalLengthIn35mmFilm: int(float(srt_data.iloc[index_export_frame]["focal_len"]) / 10) ## Short
                    }

                    # Read the existing EXIF metadata from the image file
                    exif_dict = piexif.load(str(image_path))

                    # Get the existing GPS IFD dictionary or create a new one if it doesn't exist
                    if piexif.GPSIFD in exif_dict:
                        gps_ifd = exif_dict[piexif.GPSIFD]
                    else:
                        gps_ifd = {}

                    if piexif.GPSIFD in exif_dict:
                        exif_ifd = exif_dict[piexif.ExifIFD]
                    else:
                        exif_ifd = {}

                    # Add the new GPS tag data to the GPS IFD dictionary
                    if not flag_lat_lon_captured:
                        gps_ifd.update(gps_data)
                    exif_ifd.update(exif_data)
                    exif_dict= {"0th":{}, "Exif":exif_ifd, "GPS":gps_ifd, "1st":{}, "thumbnail":None}

                    # Save the updated EXIF metadata back to the image file
                    exif_bytes = piexif.dump(exif_dict)
                    piexif.insert(exif_bytes, str(image_path))
                    
                    temp_geo_coord = np.float64(srt_data.iloc[index_export_frame][["latitude", "longitude", alt_label]].values)

                    print(movie_path_each.stem, 
                        "{:05}".format(index_export_frame),
                        "/",
                        nb_frames,
                        " lat: ", "{:<11.6f}".format(temp_geo_coord[0]),
                        "lon: ", "{:<11.6f}".format(temp_geo_coord[1]),
                        "alt: ", "{:<8.3f}".format(temp_geo_coord[2]))


    elif reference_gnss_data == 1:
            
        # format csv data
        csv_data = pd.read_csv(csv_path, encoding=csv_encoding)
        csv_data["isVideo_diff"] = csv_data["isVideo"].diff()
        print(csv_data[["datetime(utc)", "isVideo_diff"]][csv_data["isVideo_diff"] != 0])

        temp_offset_index = 0
        initial_datatime_utc = csv_data["datetime(utc)"].iloc[0]

        for i in range(len(csv_data)):
            if csv_data["datetime(utc)"].iloc[i+1] != initial_datatime_utc:
                temp_offset_index = i
                break

        csv_data["datetime(utc)_epoc"] = pd.to_datetime(csv_data["datetime(utc)"])

        for i in range(len(csv_data)):
            csv_data.loc[i, "datetime(utc)_epoc"] += offsets.Milli(((i - (temp_offset_index + 1)) % 5) * 200)

        # generate frames
        for movie_path_each in movie_path:
            # make result directory
            result_dir_path = movie_path_each.parent / "res"
            result_dir_path.mkdir(exist_ok=True)
            
            ## get basic information of the movie file
            video_info = ffmpeg.probe(movie_path_each)
            nb_frames = int(video_info["streams"][0]["nb_frames"])
            duration = float(video_info["format"]["duration"])
            temp_format = "%Y-%m-%dT%H:%M:%S.000000Z"
            start_time = datetime.datetime.strptime(video_info["format"]["tags"]["creation_time"], temp_format)
            print(duration, start_time)

            movie_cap = cv2.VideoCapture(str(movie_path_each))
            
            if end_frame_temp == 0:
                end_frame = nb_frames
            else:
                end_frame = end_frame_temp
                        
            for index_export_frame in range(start_frame, end_frame, frame_interval):

                ### calculate datetime, lat, lon, and height
                frame_time = start_time + datetime.timedelta(seconds=duration * index_export_frame / nb_frames)
                csv_corresponded_index = ((csv_data["datetime(utc)_epoc"] - frame_time) / np.timedelta64(1, 's') > 0).argmax()

                gps_location_before = csv_data[["latitude", "longitude", "altitude_above_seaLevel(meters)"]].iloc[csv_corresponded_index].values
                gps_location_after  = csv_data[["latitude", "longitude", "altitude_above_seaLevel(meters)"]].iloc[csv_corresponded_index + 1].values

                gps_location = gps_location_before + (gps_location_after - gps_location_before) \
                            * ((frame_time - csv_data["datetime(utc)_epoc"].iloc[csv_corresponded_index]) / np.timedelta64(1, 's')) \
                            / ((csv_data["datetime(utc)_epoc"].iloc[csv_corresponded_index + 1] - csv_data["datetime(utc)_epoc"].iloc[csv_corresponded_index]) / np.timedelta64(1, 's'))

                ### create image from frame
                movie_cap.set(cv2.CAP_PROP_POS_FRAMES, index_export_frame)
                ret, frame = movie_cap.read()
                if ret:
                    frame_name = movie_path_each.stem + "_" + "{:05}".format(index_export_frame) + ".jpeg"
                    image_path = result_dir_path / frame_name
                    cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 100]) 

                    # Create a new GPS tag with latitude, longitude, and altitude values
                    gps_data = {
                        piexif.GPSIFD.GPSLatitudeRef: 'N',
                        piexif.GPSIFD.GPSLatitude: _deg2rational(gps_location[0]),
                        piexif.GPSIFD.GPSLongitudeRef: 'E',
                        piexif.GPSIFD.GPSLongitude: _deg2rational(gps_location[1]),
                        piexif.GPSIFD.GPSAltitudeRef: 0,
                        piexif.GPSIFD.GPSAltitude: (int(gps_location[2] * 1000), 1000)
                    }

                    # Read the existing EXIF metadata from the image file
                    exif_dict = piexif.load(str(image_path))

                    # Get the existing GPS IFD dictionary or create a new one if it doesn't exist
                    if piexif.GPSIFD in exif_dict:
                        gps_ifd = exif_dict[piexif.GPSIFD]
                    else:
                        gps_ifd = {}

                    # Add the new GPS tag data to the GPS IFD dictionary
                    gps_ifd.update(gps_data)
                    exif_dict= {"0th":{}, "Exif":{}, "GPS":gps_ifd, "1st":{}, "thumbnail":None}

                    # Save the updated EXIF metadata back to the image file
                    exif_bytes = piexif.dump(exif_dict)
                    piexif.insert(exif_bytes, str(image_path))

                    print(movie_path_each.stem, 
                        "{:05}".format(index_export_frame),
                        "/",
                        nb_frames,
                        gps_location)








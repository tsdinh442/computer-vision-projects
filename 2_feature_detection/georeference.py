import cv2
import numpy as np

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from pyproj import Proj, transform, Geod

def process_drone(image_path):
    """
    read DJI images and extract their geodata
    below is the list of all geo data:
    YResolution
    XPComment
    XPKeywords
    GPSInfo
    GPSVersionID: b'\x02\x03\x00\x00'
    GPSLatitudeRef: N
    GPSLatitude: (30.0, 34.0, 1.025)
    GPSLongitudeRef: W
    GPSLongitude: (97.0, 39.0, 12.4735)
    GPSAltitudeRef: b'\x00'
    GPSAltitude: 344.84
    GPSStatus: A
    GPSMapDatum: WGS-84�
    ResolutionUnit
    Software
    DateTime
    ExifOffset
    ExifVersion
    ComponentsConfiguration
    ShutterSpeedValue
    DateTimeOriginal
    DateTimeDigitized
    ApertureValue
    ExposureBiasValue
    MaxApertureValue
    SubjectDistance
    MeteringMode
    LightSource
    Flash
    FocalLength
    ColorSpace
    ExifImageWidth
    SceneCaptureType
    ExifImageHeight
    Contrast
    Saturation
    Sharpness
    DeviceSettingDescription
    UniqueCameraModel
    FileSource
    ExposureTime
    ExifInteroperabilityOffset
    FNumber
    SceneType
    ExposureProgram
    CustomRendered
    ISOSpeedRatings
    ExposureMode
    FlashPixVersion
    SensitivityType
    WhiteBalance
    BodySerialNumber
    LensSpecification
    DigitalZoomRatio
    FocalLengthIn35mmFilm
    GainControl
    MakerNote

    :param image_path: drone image
    :return: gps coordinates and image coordinates of the center point ie. (lat, long), (x, y)
    """

    def Convert_2_Long_Lat(degrees, minutes, seconds, ref):
        """
        covert geodata in degrees, minutes, and seconds format to longitude, latitude format
        :param degrees:
        :param minutes:
        :param seconds:
        :param ref:
        :return: float converted value in decimal
        """
        references = {'N': 1,
                      'S': -1,
                      'E': 1,
                      'W': -1}

        # Convert to decimal degrees
        decimal_degrees = degrees + (minutes / 60.0) + (seconds / 3600.0)

        return references[ref] * decimal_degrees

    image = Image.open(image_path)

    if image:
        geodata = {}
        exif_data = image._getexif()

        if exif_data is not None:
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)

                if tag_name == 'GPSInfo':
                    for gps_tag, gps_value in value.items():
                        gps_tag_name = GPSTAGS.get(gps_tag, gps_tag)
                        # (f"{gps_tag_name}: {gps_value}")
                        geodata[gps_tag_name] = gps_value
                elif tag_name == 'DateTime':
                    geodata[tag_name] = value

        longitude = Convert_2_Long_Lat(*geodata["GPSLongitude"], geodata["GPSLongitudeRef"])
        latitude = Convert_2_Long_Lat(*geodata["GPSLatitude"], geodata["GPSLatitudeRef"])
        gps_coordinates = (latitude, longitude)

        scale_factor = 1
        new_width = int(image.width * scale_factor)
        new_height = int(image.height * scale_factor)
        new_size = (new_width, new_height)

        # Resize the image
        image = image.resize(new_size)

        w, h = image.size
        image_coordinates = (int(w / 2), int(h / 2))

        return gps_coordinates, image_coordinates, geodata['DateTime']


def process_satellite(jgw_path, gps_coord, pixel_coord=None):
    """
    given jgw file and a gps coordinate, compute matching the pixel coordinate on the satellite image
    :param jgw_path:
    :param gps_coord: tuple (lat, lon)
    :return:
    """

    def transformation_params(jgw_path):
        """

        :param jgw_path:
        :return:
        """

        # Read transformation parameters from the JGW file
        with open(jgw_path, 'r') as jgw_file:
            lines = jgw_file.readlines()
            transf_params = {
                'A': float(lines[0].strip()),  # Pixel width
                'D': float(lines[1].strip()),  # Rotation parameter
                'B': float(lines[2].strip()),  # Rotation parameter
                'E': float(lines[3].strip()),  # Pixel height
                'C': float(lines[4].strip()),  # x-coordinate of the upper-left corner of the image
                'F': float(lines[5].strip())   # y-coordinate of the upper-left corner of the image
            }

        return transf_params

    # Function to convert decimal latitude and longitude to UTM coordinates in meters
    def decimal_to_utm(lat, lon):
        """
        convert decimal to utm
        :param lat: float latitude in decimal
        :param lon: float longitude in decimal
        :return:
        """
        utm_proj = Proj(proj='utm', zone=14, datum='WGS84')
        return utm_proj(lon, lat)

    # Function to convert decimal latitude and longitude to pixel coordinates on images
    def gps_to_pixel(transformation_params, gps_coord):
        """
        convert utm coordinates to pixel coordinates on satellite image
        :param transformation_params: Dictionary containing JGW transformation parameters (A, D, B, E, C, F).
        :param gps_coord: tuple - (decimal gps lat, long)
        :return:
        """
        lat, lon = gps_coord
        utm_proj = Proj(proj='utm', zone=14, datum='WGS84')
        utm_lon, utm_lat = utm_proj(lon, lat)  # utm convention is in lon, lat order

        # Convert UTM coordinates to pixel coordinates using JGW transformation parameters
        x_pixel = int((utm_lon - transformation_params['C']) / transformation_params['A'])
        y_pixel = int((utm_lat - transformation_params['F']) / transformation_params['E'])

        return x_pixel, y_pixel

    def pixel_to_gps(transformation_params, pixel_coord):
        """
        Convert pixel coordinates to GPS coordinates.

        :param transformation_params (dict): Dictionary containing JGW transformation parameters (A, D, B, E, C, F).
        :param pixel_coord (tuple): Tuple containing x and y pixel coordinates.

        Returns: - tuple: Tuple containing latitude and longitude GPS coordinates.
        """
        x_pixel, y_pixel = pixel_coord

        # Convert pixel coordinates to UTM coordinates
        utm_lon = transformation_params['C'] + transformation_params['A'] * x_pixel
        utm_lat = transformation_params['F'] + transformation_params['E'] * y_pixel

        # Convert UTM coordinates to GPS coordinates
        utm_proj = Proj(proj='utm', zone=14, datum='WGS84')
        lon, lat = utm_proj(utm_lon, utm_lat, inverse=True)

        return lat, lon

    params = transformation_params(jgw_path)
    x, y = gps_to_pixel(params, gps_coord)
    lat, lon = None, None
    if pixel_coord:
        lat, lon = pixel_to_gps(params, pixel_coord)

    return (int(x), int(y)), (lat, lon)


def distance_per_pixel_ratio(point_1, point_2, GPS_1, GPS_2):
    """
    given 2 reference points, calculate the number pixels for each unit distance
    :param point_1: tuple (x, y) pixel coordinate of point 1, preferably the centroid of the image
    :param point_2: tuple (x, y) pixel coordinate of point 2,
    :param GPS_1: tuple (x, y) GPS coordinate of point 1
    :param GPS_2: tuple (x, y) GPS coordinate of point 2
    :return: tuple (lat ratio, long ratio)
    """
    # unpack coordinate values
    x1, y1 = point_1
    x2, y2 = point_2
    lat1, lon1 = GPS_1
    lat2, lon2 = GPS_2

    return abs((y1 - y2) / (lat1 - lat2)), abs((x1 - x2) / (lon1 - lon2))


def distance_per_pixel(gps_coord1, gps_coord2, pixel_coord1, pixel_coord2):
    """
    Calculate the ratio of distance per pixel between two GPS coordinates and their matching pixel coordinates.

    Args:
    - gps_coord1 (tuple): Tuple containing latitude and longitude of the first GPS coordinate.
    - gps_coord2 (tuple): Tuple containing latitude and longitude of the second GPS coordinate.
    - pixel_coord1 (tuple): Tuple containing x and y pixel coordinates of the first point.
    - pixel_coord2 (tuple): Tuple containing x and y pixel coordinates of the second point.

    Returns:
    - float: Ratio of distance per pixel.
    """
    # Calculate GPS distance
    geod = Geod(ellps='WGS84')
    gps_distance = geod.inv(gps_coord1[1], gps_coord1[0], gps_coord2[1], gps_coord2[0])[2]

    # Calculate pixel distance
    pixel_distance = ((pixel_coord2[0] - pixel_coord1[0]) ** 2 + (pixel_coord2[1] - pixel_coord1[1]) ** 2) ** 0.5

    # Calculate ratio of distance per pixel
    ratio_distance_per_pixel = gps_distance / pixel_distance

    return ratio_distance_per_pixel


if __name__ == "__main__":

    drone_path = "../media/collection/dji1_org.JPG"



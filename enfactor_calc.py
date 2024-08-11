import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging

# obtain the weather data at the exact location and timestamp of the ship
def get_weather_data(ship_latitude, ship_longitude, ship_timestamp, weather_df):
  gdfw_temp = weather_df.loc[weather_df.timestamp_ == ship_timestamp].copy() # select the specific timestamp for all weather stations

  timestamp_array = gdfw_temp['timestamp_'].values
  # Ensure timestamp is numeric
  if not np.issubdtype(timestamp_array.dtype, np.number):
      raise ValueError("Timestamp array must be numeric")


  # Handle potential zero variation in gust data
  if np.std(gdfw_temp['GUST'].values) < 1e-1:
      interpolated_gust_value = np.mean(gdfw_temp['GUST'].values)
  else:
      # Perform Ordinary Kriging for gust values
      OK_gust = OrdinaryKriging(gdfw_temp['lat'].values, gdfw_temp['lon'].values, gdfw_temp['GUST'].values, variogram_model='linear', verbose=False, enable_plotting=False)
      interpolated_gust_value, variance_gust = OK_gust.execute('points', np.array([ship_latitude]), np.array([ship_longitude]))

  # Handle potential zero variation in visibility data
  if np.std(gdfw_temp['VIS'].values) < 1e-1:
      interpolated_vis_value=np.mean(gdfw_temp['VIS'].values)
  else:
      # Perform Ordinary Kriging for visibility values
      OK_vis = OrdinaryKriging(gdfw_temp['lat'].values, gdfw_temp['lon'].values, gdfw_temp['VIS'].values, variogram_model='linear', verbose=False, enable_plotting=False)
      interpolated_vis_value, variance_vis = OK_vis.execute('points', np.array([ship_latitude]), np.array([ship_longitude]))

  # Handle potential zero variation in wind direction data
  if np.std(gdfw_temp['WDIRMET'].values) < 1e-1:
      interpolated_wdirmet_value=np.mean(gdfw_temp['WDIRMET'].values)
  else:
      # Perform Ordinary Kriging for mathematical wind direction values
      OK_wdirmet = OrdinaryKriging(gdfw_temp['lat'].values, gdfw_temp['lon'].values, gdfw_temp['WDIRMET'].values, variogram_model='linear', verbose=False, enable_plotting=False)
      interpolated_wdirmet_value, variance_wdirmet = OK_wdirmet.execute('points', np.array([ship_latitude]), np.array([ship_longitude]))

  weather_data_list = [interpolated_gust_value[0], interpolated_vis_value, interpolated_wdirmet_value[0]]

  return weather_data_list


# calculate enfactor for a ship given its location, timestamp and course
# also requires the weather dataset obtained after pre-processing

def get_enfactors(ship_latitude, ship_longitude, ship_timestamp, ship_course, weather_df, weather_weights_list):

  weather_data_list = get_weather_data(ship_latitude, ship_longitude, ship_timestamp, weather_df)
  gust, vis, wdirmet = weather_data_list[0], weather_data_list[1], weather_data_list[2]

  # p values for gust
  if gust <= 5.364:
    p_value_gust = 1
  elif 5.364 < gust <= 10.729:
    p_value_gust = 2
  elif 10.729 < gust <= 16.988:
    p_value_gust = 3
  elif 16.988 < gust <= 24.14:
    p_value_gust = 4
  elif 24.14 < gust:
    p_value_gust = 5

  len_gust_pvals = 5
  enfactor_gust = (weather_weights_list[0] * p_value_gust) / len_gust_pvals


  # p values for visibility
  if vis < 926:
    p_value_vis = 4
  elif 926 <= vis <= 3704:
    p_value_vis = 3
  elif 3704 <= vis <= 9260:
    p_value_vis = 2
  elif 9260 < vis:
    p_value_vis = 1

  len_vis_pvals = 4
  enfactor_vis = (weather_weights_list[1] * p_value_vis) / len_vis_pvals


  net_w_angle = ship_course - wdirmet

  # Normalize net_w_angle to be within -180 to 180 degrees
  net_w_angle = (net_w_angle + 180) % 360 - 180

  # p values for net_angle
  if -45 <= net_w_angle <= 45:
    p_value_net_w_angle = 1
  elif (-135 <= net_w_angle < -45) or (45 < net_w_angle <= 135):
    p_value_net_w_angle = 2
  elif (-180 <= net_w_angle < -135) or (135 < net_w_angle <= 180):
    p_value_net_w_angle = 3

  len_net_w_angle_pvals = 3
  enfactor_net_w_angle = (weather_weights_list[2] * p_value_net_w_angle) / len_net_w_angle_pvals


  enfactors = np.sum([enfactor_gust, enfactor_vis, enfactor_net_w_angle])
  return enfactors
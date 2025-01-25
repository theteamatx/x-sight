"""Analysis helpers script for mq."""

import datetime
import json
from urllib.parse import urlparse

from google.cloud import storage
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

datetime = datetime.datetime


def calculate_latency_per_messages(logs):
  """Calculate latency per message from a list of logs.

  Args:
      logs (list): List of dictionaries containing log data. Each log should
        include 'timestamp' and 'state'.

  Returns:
      pd.DataFrame: DataFrame with message IDs and corresponding latency in
      seconds.
  """
  df = pd.DataFrame(logs)
  df['timestamp'] = pd.to_datetime(df['timestamp'])  # Convert to datetime
  pending_times = df[df['state'] == 'pending'].set_index(
      'message_id')['timestamp']
  completed_times = df[df['state'] == 'completed'].set_index(
      'message_id')['timestamp']
  latency = completed_times - pending_times
  latency_df = latency.reset_index(name='latency')
  latency_df['latency_seconds'] = latency_df['latency'].dt.total_seconds()
  # Summary statistics
  # stats = latency_df['latency_seconds'].describe()
  # print(stats)
  return latency_df


def calculate_throughput(logs, time_interval='10S'):
  """Calculate throughput (tasks completed per time interval) from a list of logs.

  Args:
      logs (list): List of dictionaries containing log data. Each log should
        include 'timestamp' and 'state'.
      time_interval (str): Pandas offset string for time intervals (e.g., '1T'
        for 1 minute).

  Returns:
      pd.DataFrame: DataFrame with time intervals and corresponding throughput.
  """
  # Convert logs to a DataFrame
  df = pd.DataFrame(logs)

  # Ensure 'timestamp' is in datetime format
  df['timestamp'] = pd.to_datetime(df['timestamp'])

  # Filter logs for the 'completed' state
  completed_logs = df[df['state'] == 'completed']

  # Calculate throughput by resampling based on time intervals
  throughput = (
      completed_logs.set_index('timestamp').resample(time_interval).size())

  print(f'throughput => {throughput}')

  # Reset index and rename the throughput column
  return throughput.reset_index(name='throughput')


def plot_throughput_trends(throughput_df, desc=''):
  """Plots interactive throughput trends over time using Plotly.

  Args:
      throughput_df (pd.DataFrame): DataFrame with `timestamp` and `throughput`
        columns.
      desc (str): Description or title prefix for the plot.
  """
  fig = go.Figure()

  # Add throughput trace
  fig.add_trace(
      go.Scatter(
          x=throughput_df['timestamp'],
          y=throughput_df['throughput'],
          mode='lines+markers',
          marker=dict(color='green'),
          line=dict(color='green'),
          name='Throughput',
      ))

  # Update layout
  fig.update_layout(
      title=f'{desc} Plot for Throughput Trends Over Time',
      xaxis_title='Time',
      yaxis_title='Tasks Completed (Throughput)',
      template='plotly_dark',
      legend=dict(title='Legend'),
      xaxis=dict(showgrid=True),
      yaxis=dict(showgrid=True),
  )

  fig.show()


def plot_latency_trends(latency_df, desc=''):
  """Plots interactive latency trends over message IDs using Plotly.

  Args:
      latency_df (pd.DataFrame): DataFrame with `message_id` and
        `latency_seconds` columns.
      desc (str): Description or title prefix for the plot.
  """
  fig = go.Figure()

  # Add latency trace
  fig.add_trace(
      go.Scatter(
          x=latency_df['message_id'],
          y=latency_df['latency_seconds'],
          mode='lines+markers',
          marker=dict(color='blue'),
          line=dict(color='blue'),
          name='Latency',
      ))

  # Update layout
  fig.update_layout(
      title=f'{desc} Plot for Latency Trends Over Time',
      xaxis_title='Message ID',
      yaxis_title='Latency (seconds)',
      template='plotly_dark',
      legend=dict(title='Legend'),
      xaxis=dict(showgrid=True),
      yaxis=dict(showgrid=True),
  )

  fig.show()


def analyze_message_flow(logs, desc=''):
  """Analyzes the message flow from a list of logs.

  Args:
      logs (list): List of dictionaries containing log data. Each log should
        include 'timestamp' and 'state'.
      desc (str): Description or title prefix for the plot.
  """

  # Create a DataFrame from the logs
  df = pd.DataFrame(logs)

  # Convert the 'timestamp' column to datetime type
  df['timestamp'] = pd.to_datetime(df['timestamp'])

  # Sort by 'timestamp' to maintain proper order
  df = df.sort_values('timestamp')

  # Create an interactive scatter plot for the message states over time
  fig = px.line(
      df,
      x='timestamp',
      y='message_id',
      color='state',  # Color by the state to differentiate states visually
      title=desc + 'Plot for Message Transitions Over Time',
      labels={
          'timestamp': 'Time',
          'message id': 'Message_ID'
      },
      hover_data=['message_id'],
      template='plotly_dark',  # Optional: dark theme
      markers=True,
  )

  # Customize layout for better readability
  fig.update_layout(
      hovermode='closest',
      xaxis=dict(showgrid=True, title='Timestamp'),
      yaxis=dict(showgrid=True, title='State'),
      margin=dict(t=40, b=40, l=40,
                  r=40),  # Adjust margins to make space for labels
  )

  # Show the plot
  fig.show()


def analyze_lentency(logs, desc):
  latency_df = calculate_latency_per_messages(logs)
  # print(latency_df)
  # plot_latency_distribution(latency_df)
  plot_latency_trends(latency_df, desc)


def analyze_throughtput(logs, desc):
  throughput_df = calculate_throughput(logs)
  # print(throughput_df)
  plot_throughput_trends(throughput_df, desc)

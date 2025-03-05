"""Analysis helpers script for mq."""

import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

datetime = datetime.datetime


def analyze_task_logs(logs, config):
  """Analyze task logs and compute execution efficiency.

  :param logs: List of dictionaries with 'state' and 'timestamp' (YYYY-MM-DD
  HH:MM:SS)
  :param config: Dictionary containing:
                 - 'ideal_task_time': Ideal time per task in seconds (default:
                 2s per task)
                 - 'total_tasks': Total number of tasks processed (default:
                 20,000)
                 - 'workers': Number of workers processing tasks (default: 200)
  :return: Dictionary containing analysis results
  """

  # Extract config values with defaults
  ideal_task_time = config.get("ideal_task_time", 2)
  total_tasks = config.get("total_tasks", 20000)
  workers = config.get("workers", 200)

  # Convert list of dictionaries into a pandas DataFrame
  df = pd.DataFrame(logs)

  df["timestamp"] = df["timestamp"].str.replace(r"(?<=\d{2}:\d{2}:\d{2})(?!\.)",
                                                ".000000",
                                                regex=True)

  # Ensure 'timestamp' column is in datetime format
  df["timestamp"] = pd.to_datetime(df["timestamp"])

  # Find the first 'active' timestamp
  first_active = df[df["state"] == "active"]["timestamp"].min()

  # Find the last 'completed' timestamp
  last_completed = df[df["state"] == "completed"]["timestamp"].max()

  if pd.isna(first_active) or pd.isna(last_completed):
    return {"error": "Insufficient data for analysis"}

  # Compute total time taken
  total_time_taken = (last_completed - first_active).total_seconds()

  # Compute ideal time based on given task execution time
  ideal_time_taken = (
      total_tasks * ideal_task_time) / workers  # Time with perfect parallelism

  # Compute efficiency
  efficiency = ((ideal_time_taken / total_time_taken) *
                100 if total_time_taken > 0 else 0)

  # Print analysis results
  print(f"First Active Timestamp: {first_active.strftime('%Y-%m-%d %H:%M:%S')}")
  print("Last Completed Timestamp:"
        f" {last_completed.strftime('%Y-%m-%d %H:%M:%S')}")
  print(f"Total Time Taken: {total_time_taken:.2f} seconds")
  print(f"Ideal Time Taken (based on {ideal_task_time}s per task with"
        f" {workers} workers): {ideal_time_taken:.2f} seconds")
  print(f"System Efficiency: {efficiency:.2f}%")

  print(f'{"---"*10}\n\n')

  return {
      "first_active": first_active.strftime("%Y-%m-%d %H:%M:%S"),
      "last_completed": last_completed.strftime("%Y-%m-%d %H:%M:%S"),
      "total_time_taken_seconds": total_time_taken,
      "ideal_time_taken_seconds": ideal_time_taken,
      "efficiency_percentage": efficiency,
  }


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
  df["timestamp"] = df["timestamp"].str.replace(r"(?<=\d{2}:\d{2}:\d{2})(?!\.)",
                                                ".000000",
                                                regex=True)
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

  df["timestamp"] = df["timestamp"].str.replace(r"(?<=\d{2}:\d{2}:\d{2})(?!\.)",
                                                ".000000",
                                                regex=True)

  # Ensure 'timestamp' is in datetime format
  df['timestamp'] = pd.to_datetime(df['timestamp'])

  # Filter logs for the 'completed' state
  completed_logs = df[df['state'] == 'completed']

  # Calculate throughput by resampling based on time intervals
  throughput = (
      completed_logs.set_index('timestamp').resample(time_interval).size())

  # print(f'throughput => {throughput}')

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

  df["timestamp"] = df["timestamp"].str.replace(r"(?<=\d{2}:\d{2}:\d{2})(?!\.)",
                                                ".000000",
                                                regex=True)

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
      title=f'{desc} Plot for Message Transitions Over Time',
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


def analyze_full_queue_sizes(logs, desc=''):
  """Analyzes the sizes of the pending, active, and completed queues over time.

  Args:
      logs (list): List of dictionaries containing log data. Each log should
        include 'timestamp', 'state', and 'message_id'.
      desc (str): Description or title prefix for the plot.
  """

  # Create a DataFrame from the logs
  df = pd.DataFrame(logs)

  df["timestamp"] = df["timestamp"].str.replace(r"(?<=\d{2}:\d{2}:\d{2})(?!\.)",
                                                ".000000",
                                                regex=True)

  # Convert the 'timestamp' column to datetime type
  df['timestamp'] = pd.to_datetime(df['timestamp'])

  # Sort logs by timestamp to ensure proper order
  df = df.sort_values('timestamp')

  # Initialize columns for queue changes
  df['pending_change'] = df['state'].apply(lambda x: 1 if x == 'pending' else -1
                                           if x == 'active' else 0)
  df['active_change'] = df['state'].apply(lambda x: 1 if x == 'active' else -1
                                          if x == 'completed' else 0)
  df['completed_change'] = df['state'].apply(lambda x: 1
                                             if x == 'completed' else 0)

  # Calculate cumulative sizes for each queue
  df['pending_size'] = df['pending_change'].cumsum()
  df['active_size'] = df['active_change'].cumsum()
  df['completed_size'] = df['completed_change'].cumsum()

  # Create a melted DataFrame to plot all sizes in one graph
  queue_sizes = df.melt(
      id_vars=['timestamp'],
      value_vars=['pending_size', 'active_size', 'completed_size'],
      var_name='Queue',
      value_name='Size',
  )

  # Create a line plot for queue sizes over time
  fig = px.line(
      queue_sizes,
      x='timestamp',
      y='Size',
      color='Queue',  # Different colors for each queue
      title=f'{desc} Queue Sizes Over Time',
      labels={
          'timestamp': 'Time',
          'Size': 'Queue Size'
      },
      template='plotly_dark',  # Optional: dark theme
  )

  # Customize layout for better readability
  fig.update_layout(
      hovermode='closest',
      xaxis=dict(showgrid=True, title='Timestamp'),
      yaxis=dict(showgrid=True, title='Queue Size'),
      margin=dict(t=40, b=40, l=40, r=40),  # Adjust margins
      legend=dict(title='Queue',
                  orientation='h',
                  x=0.5,
                  xanchor='center',
                  y=-0.2),
  )

  # Show the plot
  fig.show()


def analyze_message_time_taken_in_state(logs, desc=''):
  """Analyzes the time taken for messages to transition between states.

  Args:
    logs: A list of dictionaries, where each dictionary represents a log entry.
      Each log entry should contain the following keys: 'timestamp',
      'message_id', 'state', and 'time_taken'.
    desc: An optional description for the plot title.

  """

  # Create a DataFrame from the logs
  df = pd.DataFrame(logs)

  df["timestamp"] = df["timestamp"].str.replace(r"(?<=\d{2}:\d{2}:\d{2})(?!\.)",
                                                ".000000",
                                                regex=True)

  # Convert the 'timestamp' column to datetime type
  df['timestamp'] = pd.to_datetime(df['timestamp'])

  # Sort logs by timestamp to ensure proper order
  df = df.sort_values('timestamp')

  fig = px.line(
      df,
      x='message_id',
      y='time_taken',
      color='state',
      markers=True,
      title=f'{desc} Message Time taken in each state',
      labels={
          'time_taken': 'Time taken (seconds)',
          'message_id': 'Message ID',
          'state': 'Queue State',
      },
      color_discrete_map={
          'PENDING': '#FFA500',  # Orange
          'ACTIVE': '1E90FF',  # Blue
          'COMPLETED': '#32CD32',  # Green
      },
      template='plotly_dark',  # Optional: dark theme
  )

  # Customize layout for better readability
  fig.update_layout(
      hovermode='closest',
      xaxis=dict(showgrid=True, title='Message ids'),
      yaxis=dict(showgrid=True, title='Time taken in seconds'),
      margin=dict(t=40, b=40, l=40, r=40),  # Adjust margins
      legend=dict(title='Queue\'s states',
                  orientation='h',
                  x=0.5,
                  xanchor='center',
                  y=-0.2),
  )

  # Show the plot
  fig.show()


def analyze_message_time_taken_with_time(logs, desc=''):
  """Analyzes message time taken with time from logs.

  Args:
    logs: A list of dictionaries, where each dictionary represents a log entry
      and contains keys like 'timestamp', 'message_id', 'state', and
      'time_taken'.
    desc: A description of the logs.

  Returns:
    None
  """
  # Create a DataFrame from the logs
  df = pd.DataFrame(logs)

  df["timestamp"] = df["timestamp"].str.replace(r"(?<=\d{2}:\d{2}:\d{2})(?!\.)",
                                                ".000000",
                                                regex=True)

  # Convert the 'timestamp' column to datetime type
  df['timestamp'] = pd.to_datetime(df['timestamp'])

  # Sort logs by timestamp to ensure proper order
  df = df.sort_values('timestamp')

  fig = px.line(
      df,
      x='timestamp',
      y='time_taken',
      color='state',
      markers=True,
      title=f'{desc} Message Time Taken with reference to timestamp',
      labels={
          'time_taken': 'Time Taken (seconds)',
          'timestamp': 'TimeStamp',
          'state': 'Queue State',
      },
      color_discrete_map={
          'PENDING': '#FFA500',  # Orange
          'ACTIVE': '1E90FF',  # Blue
          'COMPLETED': '#32CD32',  # Green
      },
      template='plotly_dark',  # Optional: dark theme
  )

  # Customize layout for better readability
  fig.update_layout(
      hovermode='closest',
      xaxis=dict(showgrid=True, title='Time'),
      yaxis=dict(showgrid=True, title='Time taken (seconds)'),
      margin=dict(t=40, b=40, l=40, r=40),  # Adjust margins
      legend=dict(
          title="Queue's states",
          orientation='h',
          x=0.5,
          xanchor='center',
          y=-0.2,
      ),
  )

  # Show the plot
  fig.show()

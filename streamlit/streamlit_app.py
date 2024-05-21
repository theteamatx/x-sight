import streamlit as st
from google.cloud import bigquery
import pandas as pd
import time
import plotly.graph_objects as go

client = bigquery.Client()

@st.cache_data(ttl=600)
def run_child_details_query(query):
    query_job = client.query(query)
    rows_raw = query_job.result()
    rows = [{row['optimizer']: row['sight_id']} for row in rows_raw]
    return rows

# Perform query.
@st.cache_data(ttl=600)
def run_query(query):
    # print('query : ', query)
    query_job = client.query(query)
    rows_raw = query_job.result()
    rows = [row["outcome_value"] for row in rows_raw]
    return rows

def get_child_details_query(super_id):
    return ('SELECT '
            '(SELECT av.value FROM UNNEST(attribute) av WHERE av.key = "optimizer") AS optimizer, '
            'link.linked_sight_id AS sight_id '
            'FROM '
            f'sight_logs.{super_id}_log '
            'WHERE '
            'sub_type = "ST_LINK";')

def get_exp_details_query(sight_log_id):
    return f'SELECT decision_outcome.outcome_value FROM `cameltrain.sight_logs.{sight_log_id}_log` WHERE decision_outcome.outcome_value IS NOT NULL ORDER BY `order`.timestamp_ns ASC;'


st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.title("Comparison study of all the optimizers")
super_id = st.text_input("Enter Parent experiment sight ID:")

if super_id:
    experiment_ids = {}
    query = get_child_details_query(super_id)
    rows = run_child_details_query(query)
    for row in rows:
      experiment_ids.update(row)

    # with st.status("Fetching data of all optimizers...", expanded=True) as status:
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    all_data = {}
    # Collect data for all experiments
    total_optimizers = len(experiment_ids)
    progress_increment = 100 / total_optimizers
    count = 1
    # with st.spinner("Fetching data..."):
    for opt, experiment_id in experiment_ids.items():
        query = get_exp_details_query(experiment_id)
        # with st.spinner(f"Fetching data for optimizer: {opt}"):
        rows = run_query(query)
        all_data[opt] = rows
        my_bar.progress(int(count * progress_increment), text=progress_text)
        count += 1

      # status.update(label="fetched all data", state="complete", expanded=False)
    time.sleep(1)
    my_bar.empty()

    # Combine data into a DataFrame
    df = pd.DataFrame(all_data)

    # Display the DataFrame
    if st.checkbox('Show raw data'):
        st.subheader('optimizer wise generated rewards over each iteration')
        st.write(df)

    st.subheader('Comparing Optimizer performance')
    fig = go.Figure()
    for column in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[column], mode='lines', name=column))

    fig.update_layout(
        title="Rewards vs Iterations (All Experiments)",
        xaxis_title="Iterations",
        yaxis_title="Rewards",
        width=1500,  # Adjust width as needed
        height=600  # Adjust height as needed
    )

    st.plotly_chart(fig)

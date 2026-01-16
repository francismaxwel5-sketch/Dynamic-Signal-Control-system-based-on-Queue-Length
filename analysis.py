import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os

# --- Configuration ---
# Instead of single file, allow two sets
STATS_FILE_1 = 'actuated_stats.xml'
TRIPINFO_FILE_1 = 'actuated_tripinfos.xml'
STATS_FILE_2 = 'static_stats.xml'
TRIPINFO_FILE_2 = 'static_tripinfos.xml'
AVG_VEHICLE_LENGTH_M = 7.5  # Fallback length (5m vehicle + 2.5m gap) if 'queueing_length' missing

# Mapping SUMO edge IDs to readable directions
INCOMING_EDGES = {
    'n_in': 'North Incoming',
    'e_in': 'East Incoming',
    's_in': 'South Incoming',
    'w_in': 'West Incoming'
}

# Mapping Lane IDs to Entry Labels
ENTRY_LANE_MAP = {
    'n_in': 'North',
    'e_in': 'East',
    's_in': 'South',
    'w_in': 'West'
}

# Route Mapping (1-12 based on your previous input)
ROUTE_MAP = {
    'r_0': '1. N to S',
    'r_1': '2. N to E',
    'r_2': '3. E to W',
    'r_3': '4. S to E',
    'r_4': '5. W to E',
    'r_5': '6. E to S',
    'r_6': '7. E to N',
    'r_7': '8. S to N',
    'r_8': '9. W to N',
    'r_9': '10. W to S',
    'r_10': '11. N to W',
    'r_11': '12. S to W',
}

# Mapping flow ID prefix to route ID (Based on morning.rou.xml flows)
# This is required because 'routeID' is missing in tripinfo.xml
# Example: motorcycleFlow10.60 corresponds to flow 'motorcycleFlow10' which uses route r_9
FLOW_TO_ROUTE_MAP = {
    # Assuming the flow number corresponds to the route index (0-11)
    'Flow1': 'r_0', 'Flow2': 'r_1', 'Flow3': 'r_2', 'Flow4': 'r_3', 
    'Flow5': 'r_4', 'Flow6': 'r_5', 'Flow7': 'r_6', 'Flow8': 'r_7',
    'Flow9': 'r_8', 'Flow10': 'r_9', 'Flow11': 'r_10', 'Flow12': 'r_11',
}

def get_route_id_from_tripinfo(trip):
    """
    Attempts to get the routeID. If missing, infers it from the trip ID based on 
    the flow naming convention (e.g., 'carFlow10.5' -> 'Flow10' -> 'r_9').
    """
    route_id = trip.get('routeID')
    if route_id:
        return route_id
    
    trip_id = trip.get('id', '')
    
    # Example trip_id: 'motorcycleFlow10.60'
    # 1. Find the flow number (e.g., '10' from 'Flow10')
    parts = trip_id.split('Flow')
    if len(parts) == 2:
        flow_number_str = parts[1].split('.')[0] # Extracts '10'
        flow_key = f'Flow{flow_number_str}'      # Reconstructs 'Flow10'
        return FLOW_TO_ROUTE_MAP.get(flow_key)
    
    return None

def parse_queue_data(xml_file):
    """Parses static_stats.xml for time-series queue data (Meters)."""
    if not os.path.exists(xml_file):
        print(f"File not found: {xml_file}")
        return pd.DataFrame()

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing {xml_file}: {e}")
        return pd.DataFrame()

    records = []

    for interval in root.findall('interval'):
        begin_time = float(interval.get('begin'))
        end_time = float(interval.get('end'))
        duration = end_time - begin_time
        
        if duration <= 0:
            continue

        for edge in interval.findall('edge'):
            edge_id = edge.get('id')
            
            if edge_id in INCOMING_EDGES:
                # 1. Try to get explicit queueing_length (meters)
                q_len_m = edge.get('queueing_length')
                
                if q_len_m is not None:
                    avg_queue_m = float(q_len_m)
                else:
                    # 2. Fallback: Calculate from waitingTime (Vehicles) * Avg Length
                    waiting_time = float(edge.get('waitingTime', 0))
                    # Queue in Vehicles = Total Waiting Time / Duration
                    avg_queue_vehs = waiting_time / duration
                    avg_queue_m = avg_queue_vehs * AVG_VEHICLE_LENGTH_M
                
                records.append({
                    'Time (min)': begin_time / 60,
                    'Direction': INCOMING_EDGES[edge_id],
                    'Queue Length (m)': avg_queue_m
                })

    return pd.DataFrame(records)

def parse_trip_data(xml_file):
    """Parses static_tripinfos.xml for global averages per route, inferring routeID if needed."""
    if not os.path.exists(xml_file):
        print(f"File not found: {xml_file}")
        return pd.DataFrame()

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing {xml_file}: {e}. Returning empty DataFrame.")
        return pd.DataFrame()

    data = defaultdict(lambda: {'total_waittime': 0.0, 'total_duration': 0.0, 'count': 0})
    processed_count = 0
    skipped_count = 0

    for trip in root.findall('tripinfo'):
        
        # New logic: Try to get route ID, either directly or by inference
        route_id = get_route_id_from_tripinfo(trip)
        
        # Check if we have a valid route ID and crucial metrics are available
        if route_id and trip.get('duration') and trip.get('waitingTime'):
            try:
                duration = float(trip.get('duration'))
                waittime = float(trip.get('waitingTime'))
            except (TypeError, ValueError):
                skipped_count += 1
                continue
            
            # Map route ID to user-friendly label
            key = ROUTE_MAP.get(route_id, route_id)
            
            data[key]['total_waittime'] += waittime
            data[key]['total_duration'] += duration
            data[key]['count'] += 1
            processed_count += 1
        else:
            skipped_count += 1


    if skipped_count > 0:
        print(f"Warning: Skipped {skipped_count} trip records due to missing data or inability to determine route.")

    results = []
    # Sort by direction number
    sorted_keys = sorted(data.keys(), key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else 99)

    for key in sorted_keys:
        val = data[key]
        if val['count'] > 0:
            avg_wait = val['total_waittime'] / val['count']
            avg_duration = val['total_duration'] / val['count']
            
            # Infer Entry Lane from label (e.g. "1. N to S" -> "North")
            entry_char = key.split('.')[1].strip().split(' ')[0] if ' to ' in key else '?'
            entry_map = {'N': 'North', 'E': 'East', 'S': 'South', 'W': 'West'}
            entry_lane = entry_map.get(entry_char, 'Unknown')

            results.append({
                'Route': key,
                'Entry_Lane': entry_lane,
                'Avg_Wait_Time (s)': avg_wait,
                'Avg_Crossing_Time (s)': avg_duration,
                'Count': val['count']
            })

    return pd.DataFrame(results)

def create_dashboard(queue_dfs, trip_dfs, labels):
    """
    Generates a combined dashboard of metrics for two datasets.
    queue_dfs, trip_dfs: list of DataFrames
    labels: list of legend labels (e.g. ['Actuated', 'Static'])
    """
    if all(df.empty for df in queue_dfs) and all(df.empty for df in trip_dfs):
        print("No data available to plot.")
        return

    sns.set_theme(style="whitegrid", palette="viridis")
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, :]) # Top: Queue Length
    ax2 = fig.add_subplot(gs[1, 0]) # Bottom Left: Waiting Time
    ax3 = fig.add_subplot(gs[1, 1]) # Bottom Right: Crossing Time

    # --- Plot 1: Queue Length over Time (Meters) ---
    colors = ['tab10', 'Set2']
    line_handles = []
    line_labels = []
    for i, queue_df in enumerate(queue_dfs):
        if not queue_df.empty:
            # Plot each direction separately for each dataset
            directions = queue_df['Direction'].unique()
            palette = sns.color_palette(colors[i], len(directions))
            for j, direction in enumerate(directions):
                df_dir = queue_df[queue_df['Direction'] == direction]
                # Plot and collect handle for legend
                handle = ax1.plot(
                    df_dir['Time (min)'],
                    df_dir['Queue Length (m)'],
                    color=palette[j],
                    label=f"{labels[i]} - {direction}",
                    linewidth=2
                )[0]
                line_handles.append(handle)
                line_labels.append(f"{labels[i]} - {direction}")

    ax1.set_title('Queue Length (Meters) over Time (1-min Intervals)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Queue Length (m)')
    max_time = max([df['Time (min)'].max() if not df.empty else 0 for df in queue_dfs])
    ax1.set_xlim(0, max_time)
    ax1.set_xlabel('Simulation Time (Minutes)')
    if line_handles:
        ax1.legend(line_handles, line_labels, loc='upper right')

    # --- Plot 2: Avg Waiting Time per Entry Lane ---
    lane_order = ['North', 'East', 'South', 'West', 'Unknown']
    bar_handles = []
    bar_labels = []
    width = 0.35
    lanes = lane_order
    x = range(len(lanes))
    for i, trip_df in enumerate(trip_dfs):
        if not trip_df.empty:
            wait_by_lane = trip_df.groupby('Entry_Lane')['Avg_Wait_Time (s)'].mean().reindex(lanes)
            # Offset bars for each dataset
            offset = [-width/2, width/2][i]
            bar = ax2.bar(
                [xi + offset for xi in x],
                wait_by_lane.values,
                width=width,
                color=sns.color_palette(colors[i])[0],
                alpha=0.7,
                label=labels[i]
            )
            bar_handles.append(bar)
            bar_labels.append(labels[i])
    ax2.set_title('Avg. Waiting Time per Entry Lane', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Time (s)')
    ax2.set_xlabel('Entry Lane')
    ax2.set_xticks(x)
    ax2.set_xticklabels(lanes)
    ax2.legend(bar_labels)

    # --- Plot 3: Crossing Time (Travel Time) per Route ---
    route_order = sorted(set(trip_dfs[0]['Route'].tolist() if not trip_dfs[0].empty else []) |
                        set(trip_dfs[1]['Route'].tolist() if not trip_dfs[1].empty else []),
                        key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else 99)
    bar_handles3 = []
    bar_labels3 = []
    x3 = range(len(route_order))
    for i, trip_df in enumerate(trip_dfs):
        if not trip_df.empty:
            crossing_by_route = trip_df.set_index('Route').reindex(route_order)['Avg_Crossing_Time (s)']
            offset = [-width/2, width/2][i]
            bar = ax3.bar(
                [xi + offset for xi in x3],
                crossing_by_route.values,
                width=width,
                color=sns.color_palette(colors[i])[2],
                alpha=0.7,
                label=labels[i]
            )
            bar_handles3.append(bar)
            bar_labels3.append(labels[i])
    ax3.set_title('Avg. Crossing Time (Travel Duration) per Route', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Time (s)')
    ax3.set_xlabel('Traffic Route Direction')
    ax3.set_xticks(x3)
    ax3.set_xticklabels(route_order, rotation=45, ha='right', fontsize=9)
    ax3.legend(bar_labels3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("--- Traffic Metrics Analysis ---")
    # 1. Parse Data
    print(f"Reading {STATS_FILE_1} and {STATS_FILE_2}...")
    q_df1 = parse_queue_data(STATS_FILE_1)
    q_df2 = parse_queue_data(STATS_FILE_2)

    print(f"Reading {TRIPINFO_FILE_1} and {TRIPINFO_FILE_2}...")
    t_df1 = parse_trip_data(TRIPINFO_FILE_1)
    t_df2 = parse_trip_data(TRIPINFO_FILE_2)

    # 2. Display Tables
    if not q_df1.empty:
        print("\n[Sample Queue Data (Meters) - Set 1]")
        # Ensure only numeric data is rounded for display
        q_display = q_df1.copy()
        q_display['Queue Length (m)'] = q_display['Queue Length (m)'].round(2)
        print(q_display.head().to_markdown(index=False)) # Use to_markdown for clean table
    if not q_df2.empty:
        print("\n[Sample Queue Data (Meters) - Set 2]")
        # Ensure only numeric data is rounded for display
        q_display = q_df2.copy()
        q_display['Queue Length (m)'] = q_display['Queue Length (m)'].round(2)
        print(q_display.head().to_markdown(index=False)) # Use to_markdown for clean table

    if not t_df1.empty:
        print("\n[Route Performance Metrics - Set 1]")
        t_display = t_df1.copy()
        t_display['Avg_Wait_Time (s)'] = t_display['Avg_Wait_Time (s)'].round(2)
        t_display['Avg_Crossing_Time (s)'] = t_display['Avg_Crossing_Time (s)'].round(2)
        print(t_display[['Route', 'Avg_Wait_Time (s)', 'Avg_Crossing_Time (s)']].to_markdown(index=False))
    if not t_df2.empty:
        print("\n[Route Performance Metrics - Set 2]")
        t_display = t_df2.copy()
        t_display['Avg_Wait_Time (s)'] = t_display['Avg_Wait_Time (s)'].round(2)
        t_display['Avg_Crossing_Time (s)'] = t_display['Avg_Crossing_Time (s)'].round(2)
        print(t_display[['Route', 'Avg_Wait_Time (s)', 'Avg_Crossing_Time (s)']].to_markdown(index=False))

    # 3. Visualize
    create_dashboard([q_df1, q_df2], [t_df1, t_df2], ['Set 1', 'Set 2'])
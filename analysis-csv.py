import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os

# --- Configuration ---
STATS_FILE = 'actuated_stats.xml'
TRIPINFO_FILE = 'actuated_tripinfos.xml'
OUTPUT_CSV_FILE = 'traffic_analysis_results.csv' # New CSV output file
AVG_VEHICLE_LENGTH_M = 7.5

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

# Route Mapping
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

FLOW_TO_ROUTE_MAP = {
    'Flow1': 'r_0', 'Flow2': 'r_1', 'Flow3': 'r_2', 'Flow4': 'r_3', 
    'Flow5': 'r_4', 'Flow6': 'r_5', 'Flow7': 'r_6', 'Flow8': 'r_7',
    'Flow9': 'r_8', 'Flow10': 'r_9', 'Flow11': 'r_10', 'Flow12': 'r_11',
}

def get_route_id_from_tripinfo(trip):
    route_id = trip.get('routeID')
    if route_id:
        return route_id
    
    trip_id = trip.get('id', '')
    parts = trip_id.split('Flow')
    if len(parts) == 2:
        flow_number_str = parts[1].split('.')[0]
        flow_key = f'Flow{flow_number_str}'
        return FLOW_TO_ROUTE_MAP.get(flow_key)
    return None

def parse_queue_data(xml_file):
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
                q_len_m = edge.get('queueing_length')
                if q_len_m is not None:
                    avg_queue_m = float(q_len_m)
                else:
                    waiting_time = float(edge.get('waitingTime', 0))
                    avg_queue_vehs = waiting_time / duration
                    avg_queue_m = avg_queue_vehs * AVG_VEHICLE_LENGTH_M
                
                records.append({
                    'Time (min)': begin_time / 60,
                    'Direction': INCOMING_EDGES[edge_id],
                    'Queue Length (m)': avg_queue_m
                })

    return pd.DataFrame(records)

def parse_trip_data(xml_file):
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
    skipped_count = 0

    for trip in root.findall('tripinfo'):
        route_id = get_route_id_from_tripinfo(trip)
        
        if route_id and trip.get('duration') and trip.get('waitingTime'):
            try:
                duration = float(trip.get('duration'))
                waittime = float(trip.get('waitingTime'))
            except (TypeError, ValueError):
                skipped_count += 1
                continue
            
            key = ROUTE_MAP.get(route_id, route_id)
            data[key]['total_waittime'] += waittime
            data[key]['total_duration'] += duration
            data[key]['count'] += 1
        else:
            skipped_count += 1

    if skipped_count > 0:
        print(f"Warning: Skipped {skipped_count} trip records.")

    results = []
    sorted_keys = sorted(data.keys(), key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else 99)

    for key in sorted_keys:
        val = data[key]
        if val['count'] > 0:
            avg_wait = val['total_waittime'] / val['count']
            avg_duration = val['total_duration'] / val['count']
            
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

def create_dashboard(queue_df, trip_df):
    if queue_df.empty and trip_df.empty:
        return

    sns.set_theme(style="whitegrid", palette="viridis")
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    if not queue_df.empty:
        sns.lineplot(data=queue_df, x='Time (min)', y='Queue Length (m)', hue='Direction', palette='tab10', linewidth=2, ax=ax1)
        ax1.set_title('Queue Length (Meters) over Time', fontsize=14, fontweight='bold')
    
    if not trip_df.empty:
        try:
            wait_by_lane = trip_df.groupby('Entry_Lane')['Avg_Wait_Time (s)'].mean().reset_index()
            sns.barplot(data=wait_by_lane, x='Entry_Lane', y='Avg_Wait_Time (s)', palette='viridis', ax=ax2)
            ax2.set_title('Avg. Waiting Time per Entry Lane', fontsize=12, fontweight='bold')
        except: pass

        try:
            sns.barplot(data=trip_df, x='Route', y='Avg_Crossing_Time (s)', palette='rocket', ax=ax3)
            ax3.set_title('Avg. Crossing Time per Route', fontsize=12, fontweight='bold')
            ax3.tick_params(axis='x', rotation=45, labelsize=9)
        except: pass

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("--- Traffic Metrics Analysis ---")
    
    print(f"Reading {STATS_FILE}...")
    q_df = parse_queue_data(STATS_FILE)
    
    print(f"Reading {TRIPINFO_FILE}...")
    t_df = parse_trip_data(TRIPINFO_FILE)

    if not q_df.empty:
        print("\n[Sample Queue Data (Meters)]")
        print(q_df.head().round(2).to_markdown(index=False))
    
    if not t_df.empty:
        print("\n[Route Performance Metrics]")
        t_display = t_df.copy()
        t_display['Avg_Wait_Time (s)'] = t_display['Avg_Wait_Time (s)'].round(2)
        t_display['Avg_Crossing_Time (s)'] = t_display['Avg_Crossing_Time (s)'].round(2)
        print(t_display[['Route', 'Avg_Wait_Time (s)', 'Avg_Crossing_Time (s)']].to_markdown(index=False))

        # --- NEW SECTION: Calculate Global Averages and Export CSV ---
        
        # 1. Calculate Weighted Global Averages (Total Time / Total Vehicles)
        # We must use weighted average because some routes have more cars than others
        total_vehicles = t_df['Count'].sum()
        
        global_avg_wait = (t_df['Avg_Wait_Time (s)'] * t_df['Count']).sum() / total_vehicles
        global_avg_cross = (t_df['Avg_Crossing_Time (s)'] * t_df['Count']).sum() / total_vehicles

        print("\n[Overall Simulation Averages]")
        print(f"Total Vehicles Processed: {total_vehicles}")
        print(f"Global Average Wait Time:     {global_avg_wait:.2f} s")
        print(f"Global Average Crossing Time: {global_avg_cross:.2f} s")

        # 2. Prepare Data for CSV Export
        # Select relevant columns
        export_df = t_df[['Route', 'Avg_Wait_Time (s)', 'Avg_Crossing_Time (s)', 'Count']].copy()
        
        # Create a Summary Row
        summary_row = pd.DataFrame([{
            'Route': 'OVERALL AVERAGE / TOTAL',
            'Avg_Wait_Time (s)': global_avg_wait,
            'Avg_Crossing_Time (s)': global_avg_cross,
            'Count': total_vehicles
        }])

        # Concatenate summary row to the bottom
        final_export = pd.concat([export_df, summary_row], ignore_index=True)

        # 3. Export to CSV
        try:
            final_export.to_csv(OUTPUT_CSV_FILE, index=False)
            print(f"\nSuccessfully exported detailed results to: {OUTPUT_CSV_FILE}")
        except Exception as e:
            print(f"\nError exporting CSV: {e}")

    else:
        print("\nTrip data could not be parsed.")

    # Visualize
    create_dashboard(q_df, t_df)
import traci
import time
import sys
import os

# --- Configuration ---
# SUMO junction ID (based on simple_4way.net.xml)
JUNCTION_ID = "C"

# Lane IDs leading up to the junction, grouped by movement (n_in, e_in, s_in, w_in)
INCOMING_LANES = {
    # Assuming one lane per incoming edge in the basic model
    "n_in": ["n_in_0"], 
    "s_in": ["s_in_0"],
    "e_in": ["e_in_0"],
    "w_in": ["w_in_0"]
}

# Map direction string to the corresponding SUMO phase states
PHASE_STATES = {
    "N": "GGGrrrrrrrrr", # North Green (Straight and Right, assuming GGG...)
    "E": "rrrGGgrrrrrr", # East Green
    "S": "rrrrrrGGgrrr", # South Green
    "W": "rrrrrrrrrGGg"  # West Green
}

# Signal Timings (as requested by user)
GREEN_DURATION = 27
YELLOW_DURATION = 3
ALL_RED_DURATION = 0 # No explicit all-red phase is used between yellow and next green

# --- Traci Control Class ---

class AdaptiveTrafficLightController:
    def __init__(self, junction_id, phase_states, green_duration, yellow_duration):
        self.junction_id = junction_id
        self.phase_states = phase_states
        self.green_duration = green_duration
        self.yellow_duration = yellow_duration
        
        # State tracking for the new logic
        self.current_schedule = [] # The order of phases (e.g., ['N', 'S', 'W', 'E']) for the current cycle
        self.schedule_index = 0    # Index of the phase currently being served in the schedule

        # Start with an initial all-red phase for setup
        traci.trafficlight.setRedYellowGreenState(self.junction_id, "rrrrrrrrrrrr")
        traci.simulationStep()
        
    def get_queue_length(self, edge_id):
        """Calculates total queue length (in vehicles) for all lanes of an incoming edge."""
        total_queue = 0
        lane_list = INCOMING_LANES.get(edge_id, [])
        for lane_id in lane_list:
            # getLastStepHaltingNumber returns the number of vehicles standing on the lane
            total_queue += traci.lane.getLastStepHaltingNumber(lane_id)
        return total_queue

    def determine_full_cycle_schedule(self):
        """
        Determines the full 4-phase cycle schedule based on queue length ranking.
        The cycle order is determined once based on the queue lengths at the start of the cycle.
        """
        
        # 1. Get current queue lengths for all four directions
        queue_data = {}
        for edge_id in INCOMING_LANES.keys():
            # Get the single character identifier (e.g., 'n_in' -> 'N')
            direction_char = edge_id.split('_')[0].upper() 
            queue_data[direction_char] = self.get_queue_length(edge_id)
        
        print(f"--- New Cycle Start (Time {traci.simulation.getTime()}s) ---")
        print(f"Current Queue lengths: {queue_data}")

        # 2. Rank directions by queue length in descending order
        # The result is a list of tuples: [('N', 15), ('S', 12), ('W', 10), ('E', 2)]
        ranked_queues = sorted(queue_data.items(), key=lambda item: item[1], reverse=True)
        
        # 3. Extract the direction order
        self.current_schedule = [direction for direction, length in ranked_queues]
        self.schedule_index = 0
        
        print(f"Next Green Cycle Order (Max Queue First): {self.current_schedule}")

    def execute_phase(self):
        """Executes the next scheduled phase in the determined cycle."""
        
        # 1. Check if a new schedule is needed (or if this is the first run)
        if self.schedule_index >= len(self.current_schedule):
            self.determine_full_cycle_schedule()
        
        # Get the direction to serve
        if not self.current_schedule:
             # Should not happen if determine_full_cycle_schedule runs correctly
             print("Error: Empty schedule. Skipping step.")
             return

        direction_to_serve = self.current_schedule[self.schedule_index]
        
        # --- Phase Execution ---

        # 1. Yellow phase for safety clearance (if moving from a previous green)
        current_state = traci.trafficlight.getRedYellowGreenState(self.junction_id)
        if current_state != "rrrrrrrrrrrr":
            # For simplicity and safety between unrelated phases, use All-Red (optional but safer)
            traci.trafficlight.setRedYellowGreenState(self.junction_id, "rrrrrrrrrrrr")
            for _ in range(ALL_RED_DURATION):
                traci.simulationStep()
        
        # 2. Implement the Green Phase (Fixed 20s)
        green_state = self.phase_states[direction_to_serve]
        traci.trafficlight.setRedYellowGreenState(self.junction_id, green_state)
        print(f"Time {traci.simulation.getTime()}s: Green for {direction_to_serve} (Duration: {self.green_duration}s)")
        for _ in range(self.green_duration):
            traci.simulationStep()
            
        # 3. Implement the Yellow Phase (Fixed 3s)
        yellow_state = green_state.replace('G', 'y')
        traci.trafficlight.setRedYellowGreenState(self.junction_id, yellow_state)
        print(f"Time {traci.simulation.getTime()}s: Yellow for {direction_to_serve} (Duration: {self.yellow_duration}s)")
        for _ in range(self.yellow_duration):
            traci.simulationStep()

        # Move to the next phase in the schedule
        self.schedule_index += 1
        
    def run(self, max_steps):
        """Main simulation loop."""
        step = 0
        while step < max_steps:
            
            # Execute the next determined green/yellow phases
            self.execute_phase()
            
            step = traci.simulation.getTime()
            
            if step >= max_steps:
                break
            
        print(f"Simulation ended at time {traci.simulation.getTime()}.")
        traci.close()
        sys.stdout.flush()

def start_sumo(sumocfg_file, use_gui=False):
    """Starts SUMO and returns the TraCI connection status."""
    if 'SUMO_HOME' not in os.environ:
        sys.exit("Please declare the environment variable 'SUMO_HOME'")
    
    try:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
        
        sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo-gui') if use_gui else os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo')
        
        # FIX: Removed unrecognized option '--step-method duration'
        sumo_cmd = [sumo_binary, "-c", sumocfg_file, "--step-length", "1.0"] 

        print(f"Starting SUMO with command: {sumo_cmd}")
        traci.start(sumo_cmd)
        return True
    except Exception as e:
        print(f"Error starting SUMO or connecting to TraCI: {e}")
        return False

if __name__ == "__main__":
    SUMOCFG_FILE = 'simple_actuated_simulation.sumocfg'
    
    # 1. Start SUMO
    if start_sumo(SUMOCFG_FILE, use_gui=False): # Set to True to debug with GUI
        # 2. Run the controller
        controller = AdaptiveTrafficLightController(
            JUNCTION_ID,
            PHASE_STATES,
            GREEN_DURATION,
            YELLOW_DURATION
        )
        
        # Run for the duration defined in the .sumocfg file (3600s)
        controller.run(max_steps=3600)
    else:
        print("Failed to run simulation.")
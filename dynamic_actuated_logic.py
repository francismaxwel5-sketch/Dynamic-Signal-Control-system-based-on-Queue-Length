import traci
import time
import sys
import os

# --- Configuration ---
JUNCTION_ID = "C"

# Lane IDs leading up to the junction
INCOMING_LANES = {
    "n_in": ["n_in_0"], 
    "s_in": ["s_in_0"],
    "e_in": ["e_in_0"],
    "w_in": ["w_in_0"]
}

# MAPPING CORRECTION
# Based on incLanes="s_in_0 e_in_0 n_in_0 w_in_0"
# Indices 0-2: South
# Indices 3-5: East
# Indices 6-8: North
# Indices 9-11: West

PHASE_STATES = {
    "S": "GGGrrrrrrrrr",  # South is the first group (Indices 0-2)
    "E": "rrrGGgrrrrrr",  # East is the second group (Indices 3-5)
    "N": "rrrrrrGGgrrr",  # North is the third group (Indices 6-8)
    "W": "rrrrrrrrrGGg"   # West is the fourth group (Indices 9-11)
}

# --- Dynamic Logic Constraints ---
MIN_GREEN_TIME = 5      # Seconds
MAX_GREEN_TIME = 40      # Seconds
MAX_RED_WAIT_TIME = 90  # If a lane waits this long, it gets priority regardless of queue size
TIME_PER_VEHICLE = 3.5  # 2.5 seconds of green time allocated per vehicle in queue
YELLOW_DURATION = 3
ALL_RED_DURATION = 0

class DynamicTrafficLightController:
    def __init__(self, junction_id, phase_states):
        self.junction_id = junction_id
        self.phase_states = phase_states
        
        # Scheduler state
        self.current_schedule = [] 
        # Note: self.schedule_index is no longer needed with the refactored loop structure.
        
        # Track when each direction was last served (Green end time)
        # Initialize with 0
        self.last_served_time = {
            "N": 0, "E": 0, "S": 0, "W": 0
        }

        # Initialize Traffic Light
        traci.trafficlight.setRedYellowGreenState(self.junction_id, "rrrrrrrrrrrr")
        traci.simulationStep()
        
    def get_queue_length(self, edge_id):
        """Calculates total queue length (in vehicles) for all lanes of an incoming edge."""
        total_queue = 0
        lane_list = INCOMING_LANES.get(edge_id, [])
        for lane_id in lane_list:
            # getLastStepHaltingNumber counts vehicles that have been waiting for > 1s
            total_queue += traci.lane.getLastStepHaltingNumber(lane_id)
        return total_queue

    def calculate_dynamic_duration(self, queue_length):
        """
        Calculates green time based on queue length, bounded by Min/Max constraints.
        Formula: Time = Queue * TimePerVehicle
        """
        calculated_time = queue_length * TIME_PER_VEHICLE
        
        # Apply Constraints: Min >= Time >= Max
        final_time = max(MIN_GREEN_TIME, min(calculated_time, MAX_GREEN_TIME))
        return int(final_time)

    def determine_priority_schedule(self):
        """
        Determines the cycle order and durations based on:
        1. Max Red Light Rule (Starvation check)
        2. Longest Queue Rule
        """
        current_sim_time = traci.simulation.getTime()
        
        starving_lanes = []
        normal_lanes = []

        # 1. Gather Data and Classify Lanes
        print(f"\n--- Decision Logic (Time {current_sim_time}) ---")
        for edge_id in INCOMING_LANES.keys():
            direction_char = edge_id.split('_')[0].upper()
            queue_len = self.get_queue_length(edge_id)
            
            # Check how long it has been since this lane was last served
            wait_time = current_sim_time - self.last_served_time[direction_char]
            
            entry = {
                "dir": direction_char,
                "queue": queue_len,
                "wait": wait_time
            }
            
            # 2. Check Max Red Light Constraint
            if wait_time > MAX_RED_WAIT_TIME:
                print(f"(!) CRITICAL: {direction_char} has waited {wait_time:.1f}s (Max allowed: {MAX_RED_WAIT_TIME}s). Promoting.")
                starving_lanes.append(entry)
            else:
                normal_lanes.append(entry)

        # 3. Sort Lanes
        # Starving lanes: Sort by wait time (longest wait first)
        starving_lanes.sort(key=lambda x: x['wait'], reverse=True)
        
        # Normal lanes: Sort by queue length (longest queue first)
        normal_lanes.sort(key=lambda x: x['queue'], reverse=True)
        
        # Combine: Starving lanes get priority
        full_order = starving_lanes + normal_lanes
        
        # 4. Build the Schedule Tuple: (Direction, Duration)
        self.current_schedule = []
        for item in full_order:
            # Duration calculation is based on the queue size at the time of decision.
            duration = self.calculate_dynamic_duration(item['queue'])
            self.current_schedule.append((item['dir'], duration))
            
        print(f"Queue Status: {[(x['dir'], x['queue']) for x in full_order]}")
        print(f"New Schedule: {self.current_schedule}")

    def _execute_single_phase(self, direction, duration):
        """Executes a single Green-Yellow phase transition."""
        
        # --- Safety Handling (Inter-phase All-Red) ---
        # Ensures that a brief all-red phase exists between two competing greens
        current_state = traci.trafficlight.getRedYellowGreenState(self.junction_id)
        if current_state != "rrrrrrrrrrrr":
            traci.trafficlight.setRedYellowGreenState(self.junction_id, "rrrrrrrrrrrr")
            for _ in range(ALL_RED_DURATION):
                traci.simulationStep()
        
        # --- GREEN PHASE ---
        green_state = self.phase_states[direction]
        traci.trafficlight.setRedYellowGreenState(self.junction_id, green_state)
        
        print(f" > Green for {direction}: {duration}s")
        
        # Run the green phase for the determined duration
        for _ in range(duration):
            traci.simulationStep()
            
        # Update Last Served Time (We mark the end of the green phase as the last served time)
        self.last_served_time[direction] = traci.simulation.getTime()
            
        # --- YELLOW PHASE ---
        yellow_state = green_state.replace('G', 'y')
        traci.trafficlight.setRedYellowGreenState(self.junction_id, yellow_state)
        
        # Run the yellow phase for its fixed duration
        for _ in range(YELLOW_DURATION):
            traci.simulationStep()

    def execute_cycle(self):
        """
        Executes a full traffic light cycle (N, E, S, W in some order).
        A new scheduling decision is made *before* the cycle starts and is fixed for the duration.
        """
        
        # 1. DECISION: Determine the priority and duration for all phases in the upcoming cycle
        self.determine_priority_schedule()
        
        # 2. EXECUTION: Execute the full schedule sequentially using the fixed durations
        for direction, duration in self.current_schedule:
            self._execute_single_phase(direction, duration)

        # 3. CLEANUP: Clear the schedule for the next cycle
        self.current_schedule = []


    def run(self, max_steps):
        step = 0
        while step < max_steps:
            # The run loop now explicitly calls a full cycle execution
            self.execute_cycle() 
            step = traci.simulation.getTime()
            if step >= max_steps:
                break
        
        print(f"Simulation ended at time {traci.simulation.getTime()}.")
        traci.close()
        sys.stdout.flush()

def start_sumo(sumocfg_file, use_gui=False):
    if 'SUMO_HOME' not in os.environ:
        sys.exit("Please declare the environment variable 'SUMO_HOME'")
    
    try:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
        sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo-gui') if use_gui else os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo')
        sumo_cmd = [sumo_binary, "-c", sumocfg_file, "--step-length", "1.0"] 
        traci.start(sumo_cmd)
        return True
    except Exception as e:
        print(f"Error starting SUMO: {e}")
        return False

if __name__ == "__main__":
    SUMOCFG_FILE = 'simple_actuated_simulation.sumocfg'
    
    # Run logic
    if start_sumo(SUMOCFG_FILE, use_gui=True): # Change to True to see GUI
        controller = DynamicTrafficLightController(JUNCTION_ID, PHASE_STATES)
        controller.run(max_steps=3600)
    else:
        print("Failed to run simulation.")
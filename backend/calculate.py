import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import math

TMARadius = 40  # 40 nautical miles
firboundaries_file = "Boundaries/firboundaries_SIN.csv"
customboundaries_file = "Boundaries/custom_boundary.csv"

required_columns = [
    "Callsign",
    "Time",
    "ReferenceTime",
    "Speed",
    "Distance",
    "Aircraft Type",
    "Altitude",
    "Latitude",
    "Longitude"
]

###### BADA Fuel Calculation Functions ######
Cv_min = 1.3  # provided in manual
p0 = 1.01325 * 10**5
rho0 = 1.225
# a0 = 340.294
T0 = 288.15
R = 287.05287
g0 = 9.80665
bT = -0.0065
Hp_trop = 11000
p_trop = p0 * ((T0 + bT*11000)/T0)**(-g0/(bT*R))
T_trop = T0 + bT*Hp_trop

# Load BADA coefficients and holding fuel data
TMA_fuel_coeff = pd.read_csv('server/BADA_fuel_coeff.csv')
holding_data = pd.read_csv('server/BADA_fuel_coeff.csv').iloc[:, [0, 28]]  # columns 1 and 29 (0-indexed), Aircraft and Type

# NATS fuel coefficients for holding
holding_fuel = pd.DataFrame({
    'Type': ["Jet_Light", "Jet_Medium", "Jet_Upper_Medium", "Jet_Heavy", "Jet_4_Heavy", "Turboprop_Medium", "Turboprop_Heavy"],
    'Fuel': [38, 61, 94, 148, 275, 10, 17]
})

holding_fuel['Fuel'] = holding_fuel['Fuel'] / 60
holding_fuel = holding_data.merge(holding_fuel, on='Type', how='left')

##### The end goal is to calculate fuel flow (kg/min) by using inputs: v (true airspeed, knots) and Hp (altitude, feet)
##### We also need to specify the aircraft type
# This function takes the aircraft code and finds it in the TMA_fuel_coeff file,
# then assigns the value of all coefficients for that aircraft to the coefficient name, in the global environment to use in other functions
# always run this function first 

def get_aircraft_coeffs(aircraft):
    # Find the row where Aircraft column matches the input
    row = TMA_fuel_coeff[TMA_fuel_coeff['Aircraft'] == aircraft]
    
    if row.empty:
        raise ValueError(f"Aircraft {aircraft} not found in TMA_fuel_coeff")
    
    # Get the first (and should be only) matching row
    row = row.iloc[0]
    
    # Loop through all columns except 'Aircraft' and create global variables
    for col in TMA_fuel_coeff.columns[1:]:  # Skip first column (Aircraft)
        globals()[col] = row[col]

def knot_to_m_s(v, unit):
    if unit == "ms":
        v_return = v * 0.514444
    elif unit == "knots":
        v_return = v * 1.94384  # in knots
    return v_return

def Mach_to_TAS(M, Hp):
    k = 1.4
    temp = get_temp(Hp)
    v_tas = M * (k * R * temp) ** 0.5
    v_tas_kt = knot_to_m_s(v_tas, "knots")
    
    return v_tas_kt

def TAS_to_CAS(Hp, TAS):
    p = get_p(Hp)
    rho = get_rho(p, Hp)
    
    k = 1.4
    mu = (k - 1) / k
    
    TAS = knot_to_m_s(TAS, "ms")
    
    a = (1 + (mu / 2) * (rho / p) * TAS ** 2) ** (1 / mu)
    b = (1 + p / p0 * (a - 1)) ** mu
    c = ((2 / mu) * (p0 / rho0) * (b - 1)) ** 0.5
    
    return knot_to_m_s(c, "knots")

def CAS_to_TAS(Hp, CAS):
    p = get_p(Hp)
    rho = get_rho(p, Hp)
    
    # gamma here is the ratio of specific heats in air, which is 1.4
    k = 1.4
    mu = (k - 1) / k
    
    CAS = knot_to_m_s(CAS, "ms")
    
    # a,b,c are components of the equation
    a = (1 + (mu / 2) * (rho0 / p0) * CAS ** 2) ** (1 / mu)
    b = (1 + p0 / p * (a - 1)) ** mu
    c = ((2 / mu) * (p / rho) * (b - 1)) ** 0.5
    return knot_to_m_s(c, "knots")

def get_temp(Hp_ft):
    # assuming ISA conditions
    Hp_m = Hp_ft / 3.28084
    
    if Hp_m <= Hp_trop:
        temp = T0 + bT * Hp_m
    else:
        temp = T_trop
    
    return temp

def get_p(Hp):
    Hp_m = Hp / 3.28084
    temp = get_temp(Hp)
    if Hp_m >= 11000:
        p = p_trop * math.exp(-g0 / (R * T_trop) * (Hp_m - Hp_trop))
    else:
        p = p0 * (temp / T0) ** (-g0 / (bT * R))
    
    return p

def get_rho(p, Hp):
    temp = get_temp(Hp)
    rho = p / (R * temp)
    
    return rho

def get_config(v_cas, Hp, phase):
    
    if phase == "ascent":
        if Hp <= 400:
            config = "takeoff"
        elif Hp <= 2000:
            config = "init climb"
        else:
            config = "cruise"
    
    elif phase == "descent":
        if (Hp >= 8000) or ((Hp < 8000) and (v_cas >= (Cv_min * CR_Vstall + 10))):
            config = "cruise"
        elif ((Hp < 3000) and (v_cas >= (Cv_min * AP_Vstall + 10)) and (v_cas <= (Cv_min * CR_Vstall + 10))) or ((Hp >= 3000) and (v_cas <= (Cv_min * CR_Vstall + 10))):
            config = "approach"
        elif (Hp < 3000) and (v_cas < (Cv_min * AP_Vstall + 10)):
            config = "landing"
        else:
            print("error in getting config")
    
    else:
        config = "cruise"
    
    return config

def get_Thr_mc_ISA(C_Tc_1, C_Tc_2, C_Tc_3, Hp, v):
    if Engines_type == "Jet":
        Thr_mc_ISA = C_Tc_1 * (1 - (Hp / C_Tc_2) + (C_Tc_3 * Hp ** 2))
    elif Engines_type == "Turboprop":
        Thr_mc_ISA = C_Tc_1 / v * (1 - (Hp / C_Tc_2)) + C_Tc_3
    elif Engines_type == "Piston" or Engines_type == "Electric":
        Thr_mc_ISA = C_Tc_1 * (1 - (Hp / C_Tc_2)) + C_Tc_3 / v
    
    return Thr_mc_ISA

def get_Thr_mc(Thr_mc_ISA, C_Tc_4, C_Tc_5, Hp):
    
    delta_T = -0.0065 * (Hp / 3.28084)  # -0.0065 is a standard temp lapse with alt value (lambda), 3.281 is to convert feet to meters
    
    T_eff = delta_T - C_Tc_4
    
    if T_eff < 0:  # bounds in BADA
        T_eff = 0
    elif (T_eff * C_Tc_5) > 0.4:
        T_eff = 0.4 / C_Tc_5
    else:
        T_eff = T_eff
    
    if C_Tc_5 < 0:  # bounds in BADA
        C_Tc_5 = 0
    
    Thr_mc = Thr_mc_ISA * (1 - (C_Tc_5 * T_eff))
    return Thr_mc

def get_Thr_ARR(Thr_mc, Hp_des, C_Tdes_high, C_Tdes_app, C_Tdes_ld, Hp, config):
    Cv_min = 1.3  # provided in manual
    
    if Hp > Hp_des:
        Thr_ARR = C_Tdes_high * Thr_mc
        # print("high")
    elif Hp <= Hp_des:
        
        # modified to be more in line with descent phases
        if config == "cruise":
            Thr_ARR = C_Tdes_low * Thr_mc   
            # print("cruise,low")
        elif config == "approach":
            Thr_ARR = C_Tdes_app * Thr_mc
            # print("approach")
        elif config == "landing":
            Thr_ARR = C_Tdes_ld * Thr_mc
            # print("landing")
        else:
            print("error in getting thrust during descent")
            print(["Hp: ", Hp, " HP_des: ", Hp_des, " config: ", config])
    
    return Thr_ARR

def get_drag(Hp, v_tas):
    import math
    
    m = Mass_tons * 1000
    p = get_p(Hp)
    rho = get_rho(p, Hp)
    v = knot_to_m_s(v_tas, "ms")  # v in m/s
    # making an assumption here that the bank angle is 0 degrees despite the BADA paper saying nominal is 30 degrees
    # now assuming bank angle 30 if during 'holding'
    CL = (2 * m * g0 / (rho * (v ** 2) * S)) if Hp > 12000 else (2 * m * g0 / (rho * (v ** 2) * S * math.sqrt(3) / 2))
    CD = CR_CD0 + CR_CD2 * (CL ** 2)
    
    D = CD * rho * (v ** 2) * S * 0.5
    return D

def get_tsfc(C_f1, C_f2, v):
    if Engines_type == "Jet":
        tsfc = C_f1 * (1 + v / C_f2)
    elif Engines_type == "Turboprop":
        tsfc = C_f1 * (1 - v / C_f2) * (v / 1000)
    
    return tsfc

def get_f_nom(v, Thr):
    if Engines_type == "Piston" or Engines_type == "Electric":
        f_nom = C_f1
    else:
        f_nom = get_tsfc(C_f1, C_f2, v) * Thr
    
    return f_nom

def get_f_min(C_f3, C_f4, Hp):
    if Engines_type == "Piston" or Engines_type == "Electric":
        f_min = C_f3
    else:
        f_min = C_f3 * (1 - Hp / C_f4)
    
    return f_min

def get_f_cr(v, thr, C_fcr):
    if Engines_type == "Piston" or Engines_type == "Electric":
        f_cr = C_f1 * C_fcr
    else:
        f_cr = get_tsfc(C_f1, C_f2, v) * thr * C_fcr
    
    return f_cr

def get_fuel_flow(aircraft, phase, v, Hp):
    
    get_aircraft_coeffs(aircraft)
    
    Thr_mc_ISA = get_Thr_mc_ISA(C_Tc_1, C_Tc_2, C_Tc_3, Hp, v)
    Thr_mc = get_Thr_mc(Thr_mc_ISA, C_Tc_4, C_Tc_5, Hp)
    
    v_cas = TAS_to_CAS(Hp, v)
    
    config = get_config(v_cas, Hp, phase)
    
    # Speed cap is NOT implemented with this part commented out
    # v = get_v(v, Hp, config)
    # v_cas = TAS_to_CAS(Hp, v)  # ensure v_cas is based on new minimum v
    
    # Thrust calculations depending on phase
    if phase == 'descent':
        Thr_ARR = get_Thr_ARR(Thr_mc, Hp_des, C_Tdes_high, C_Tdes_app, C_Tdes_ld, Hp, config)
        Thr = Thr_ARR
    elif phase == 'ascent':
        Thr = Thr_mc
    else:
        Thr = min(get_drag(Hp, v), 0.95 * Thr_mc)  # Maximum amount of thrust available during cruise, otherwise Thr = Drag
    
    # print(Thr)
    Thr = Thr / 1000  # Thrust is in N, but tsfc is in kN
    
    # tsfc = get_tsfc(C_f1, C_f2, v)
    f_nom = get_f_nom(v, Thr)
    f_min = get_f_min(C_f3, C_f4, Hp)
    
    # Fuel calculations depending on phase & config
    if phase == 'descent':
        if config == "approach" or config == "landing":
            fuel_flow = max(f_nom, f_min)
        elif config == "cruise":
            fuel_flow = f_min
    elif phase == 'ascent':
        fuel_flow = f_nom
    else:
        fuel_flow = get_f_cr(v, Thr, C_fcr)
    
    # print(["config:", config, "Thr", Thr*1000, "Drag", get_drag(Hp, v)])
    return [fuel_flow, v_cas, config]


### FUEL FLOW LOOP ###
def get_phase(alt, time):
    phase_vector = []
    # Convert to numpy arrays for vectorized operations
    alt = np.array(alt)
    time = np.array(time)
    alt_diff = alt[1:] - alt[:-1]
    time_diff = time[1:] - time[:-1]
    
    # Convert Timedelta objects to seconds
    time_diff_seconds = np.array([dt.total_seconds() for dt in time_diff])
    
    # Calculate rate of climb (units: altitude_units per second)
    roc = alt_diff / time_diff_seconds    
    
    for x in range(len(alt) - 1):
        if x > 0:  # x > 1 in R becomes x > 0 in Python (0-indexed)
            if roc[x] > 5 or (roc[x] > 0 and phase_vector[x-1] == "ascent"):
                phase = "ascent"
            elif roc[x] < (-5) or (roc[x] < 0 and phase_vector[x-1] == "descent") or (alt[x] <= 32000 and phase_vector[x-1] == "descent"):
                phase = "descent"
            else:
                phase = "cruise"
        else:
            if roc[x] > 5:
                phase = "ascent"
            elif roc[x] < (-5):
                phase = "descent"
            else:
                phase = "cruise"
        
        # print(phase)
        phase_vector.append(phase)
    
    # last point gets the same phase as the last segment.
    phase_vector.append(phase_vector[-1])
    return phase_vector

def get_fuel_flow_multi(ac_type, tas, alt, time, lat, lon, holdmode):
    fuel_flow_cumulative = 0
    
    # Initialize lists to store results
    xfuel = []
    xfuel_over_time = []
    xfuel_cumulative = []
    xv_cas = []
    xphase = []
    xconfig = []
    xtime = []
    
    # print("getting phases")
    phase = get_phase(alt, time)
    hold = get_holding(lat, lon, phase, alt)
    
    for x in range(len(tas)):
        if hold[x] == 0 or holdmode == "BADA":
            fuel_flowEX = get_fuel_flow(ac_type[x], phase[x], tas[x], alt[x])  # original is in kg/min, but the times are recorded in seconds
            
            # print(fuel_flowEX)
            fuel_flow = float(fuel_flowEX[0]) / 60
            v_cas = fuel_flowEX[1]
            config = fuel_flowEX[2]
            time_at_point = time[x]
            phase2 = phase[x]
            
            if x > 0:
                # changed from no int to have int on line below
                fuel_flow_over_time = (fuel_flow + xfuel[x-1]) / 2 * (int(time_at_point.timestamp()) - int(time[x-1].timestamp()))  # Area under curve for that section
                
                fuel_flow_cumulative = fuel_flow_cumulative + fuel_flow_over_time
                if hold[x] == 1:
                    phase2 = "hold"
                
                xfuel.append(fuel_flow)
                xfuel_over_time.append(fuel_flow_over_time)
                xfuel_cumulative.append(fuel_flow_cumulative)
                xv_cas.append(v_cas)
                xphase.append(phase2)
                xconfig.append(config)
                xtime.append(str(time_at_point))
            else:
                xfuel.append(fuel_flow)
                xfuel_over_time.append(0)
                xfuel_cumulative.append(0)
                xv_cas.append(v_cas)
                xphase.append(phase2)
                xconfig.append(config)
                xtime.append(str(time_at_point))
                
        else:  # if in holding
            # Find fuel flow from holding_fuel DataFrame
            matching_rows = holding_fuel[holding_fuel['Aircraft'] == ac_type[x]]
            fuel_flow = matching_rows['Fuel'].iloc[0] if not matching_rows.empty else None
            
            # print(fuel_flow)
            # If it doesn't match anything in the table just... default back to regular phase behaviour, but still mark it as holding so it doesn't mess up the colours
            if holdmode == "BADA_Cruise":
                # print("holding and calculating with BADA cruise")
                fuel_flowEX = get_fuel_flow(ac_type[x], "cruise", tas[x], alt[x])
                fuel_flow = float(fuel_flowEX[0]) / 60
                v_cas = fuel_flowEX[1]
                config = fuel_flowEX[2]
            elif fuel_flow is not None and not pd.isna(fuel_flow):
                # print("NATS calculation")
                v_cas = TAS_to_CAS(alt[x], tas[x])
                config = "hold"
            else:
                # print("holding but calculating with phase")
                fuel_flowEX = get_fuel_flow(ac_type[x], phase[x], tas[x], alt[x])  # original is in kg/min, but the times are recorded in seconds
                # print(fuel_flowEX)
                fuel_flow = float(fuel_flowEX[0]) / 60
                v_cas = fuel_flowEX[1]
                config = fuel_flowEX[2]
            
            time_at_point = time[x]
            fuel_flow_over_time = (fuel_flow + xfuel[x-1]) / 2 * (int(time_at_point.timestamp()) - int(time[x-1].timestamp()))  # Area under curve for that section
            fuel_flow_cumulative = fuel_flow_cumulative + fuel_flow_over_time
            
            xfuel.append(fuel_flow)
            xfuel_over_time.append(fuel_flow_over_time)
            xfuel_cumulative.append(fuel_flow_cumulative)
            xv_cas.append(v_cas)
            xphase.append("hold")
            xconfig.append(config)
            xtime.append(str(time_at_point))
    
    # Create DataFrame
    fuel_data = pd.DataFrame({
        'fuel_at_point': xfuel,
        'fuel_flow_over_time': xfuel_over_time,
        'fuel_cumulative': xfuel_cumulative,
        'v_cas': xv_cas,
        'phase': xphase,
        'config': xconfig,
        'time': xtime
    })
    
    return fuel_data

#### HOLDING IDENTIFICATION FUNCTION ####
def bearing(point1, point2):
    """
    Calculate the initial bearing from point1 to point2
    
    Args:
        point1: [longitude, latitude] of first point
        point2: [longitude, latitude] of second point  
    
    Returns:
        Initial bearing in degrees (-180 to 180)
    """
    lng1, lat1 = point1
    lng2, lat2 = point2
    
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    lng1_rad = math.radians(lng1)
    lng2_rad = math.radians(lng2)
    
    # Calculate bearing
    dlon = lng2_rad - lng1_rad
    y = math.sin(dlon) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
    
    bearing_rad = math.atan2(y, x)
    bearing_deg = math.degrees(bearing_rad)
    
    return bearing_deg

def get_bearing(lng1, lat1, lng2, lat2):
    x = bearing([lng1, lat1], [lng2, lat2])
    
    if x < 0:
        x = 360 + x
    
    return x

def get_holding(Latitude, Longitude, Phase, Altitude):
    bearing = [np.nan] * (len(Latitude) - 1)
    
    for i in range(len(Latitude) - 1):
        bearing[i] = get_bearing(Longitude[i], Latitude[i], Longitude[i+1], Latitude[i+1])
    
    bearing_diff = [np.nan] * (len(Latitude) - 1)
    
    for i in range(1, len(Latitude) - 1):
        bearing_diff[i] = abs(bearing[i] - bearing[i-1])
    
    holding = [0] * len(Latitude)
    
    # Filter out NaN values for max calculation
    bearing_diff_clean = [x for x in bearing_diff if not np.isnan(x)]
    
    if len(bearing_diff_clean) > 0 and max(bearing_diff_clean) >= 30:
        
        change_rows = [i for i, x in enumerate(bearing_diff) if not np.isnan(x) and x >= 30]
        
        conseq_5_rows = []
        
        for row in change_rows:
            score = 1
            
            for add in range(1, 5):  # 1 to 4
                test = row + add
                if (test in change_rows and 
                    test < len(Phase) and 
                    Phase[test] != "ascent" and 
                    Altitude[test] >= 8000):
                    score = score + 1
            
            if score >= 4:
                conseq_5_rows.append(row)
        
        if len(conseq_5_rows) > 0:
            
            between_row = []
            for i in range(1, len(conseq_5_rows)):
                between_row.append(conseq_5_rows[i] - conseq_5_rows[i-1])
            
            crossoveridx = [i for i, x in enumerate(between_row) if x > 5]
            
            # In case there are two or more holding stacks/loops
            if len(crossoveridx) > 0:
                crossoveridx = [0] + [x + 1 for x in crossoveridx] + [len(conseq_5_rows)]
                
                for i in range(len(crossoveridx) - 1):
                    first_row = conseq_5_rows[crossoveridx[i]]
                    last_row = conseq_5_rows[crossoveridx[i+1] - 1] + 4
                    
                    # Ensure we don't go out of bounds
                    last_row = min(last_row, len(holding) - 1)
                    
                    for j in range(first_row, last_row + 1):
                        if j < len(holding):
                            holding[j] = 1
                    
                    # Try to filter out flights that curve but do not turn in a loop
                    bearing_slice = [bearing[k] for k in range(first_row, min(last_row + 1, len(bearing)))]
                    if bearing_slice:
                        avgbearing = np.mean(bearing_slice)
                        
                        if all(abs(b - avgbearing) < 90 for b in bearing_slice if not np.isnan(b)):
                            for j in range(first_row, last_row + 1):
                                if j < len(holding):
                                    holding[j] = 0
            else:
                first_row = conseq_5_rows[0]
                last_row = min(max(conseq_5_rows) + 4, len(holding) - 1)
                
                for j in range(first_row, last_row + 1):
                    holding[j] = 1
                
                # Try to filter out flights that curve but do not turn in a loop
                bearing_slice = [bearing[k] for k in range(first_row, min(last_row + 1, len(bearing)))]
                if bearing_slice:
                    avgbearing = np.mean(bearing_slice)
                    
                    if all(abs(b - avgbearing) < 90 for b in bearing_slice if not np.isnan(b)):
                        for j in range(first_row, last_row + 1):
                            holding[j] = 0
    
    return holding

### CALCULATOR FILTER BASED ON BOUNDARIES ###
def point_in_polygon(point_lon, point_lat, polygon_coords):
    """
    Ray casting algorithm to determine if point is inside polygon
    polygon_coords should be list of [lon, lat] pairs
    """
    x, y = point_lon, point_lat  # Our test point
    n = len(polygon_coords)      # Number of polygon vertices
    inside = False               # Start assuming we're outside
    
    p1x, p1y = polygon_coords[0] # First vertex of polygon
    
    # Check each edge of the polygon
    for i in range(1, n + 1):
        p2x, p2y = polygon_coords[i % n]  # Next vertex (wraps around to start)
        
        # Check if our horizontal ray intersects this edge
        if y > min(p1y, p2y):           # Point is above the lower vertex
            if y <= max(p1y, p2y):      # Point is below (or at) the higher vertex
                if x <= max(p1x, p2x):  # Point is to the left of the rightmost vertex
                    
                    # Calculate where our ray intersects the edge
                    if p1y != p2y:  # Edge is not horizontal
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    
                    # If edge is vertical OR intersection is to our right
                    if p1x == p2x or x <= xinters:
                        inside = not inside  # Flip our inside/outside status
        
        p1x, p1y = p2x, p2y  # Move to next edge
    
    return inside

def extract_polygons(boundaries_df):
    """Convert coordinate points to polygon boundary"""
    # Extract coordinates as list of [lon, lat] pairs
    coords = boundaries_df[['longitude', 'latitude']].values.tolist()
    return coords

def get_boundaries(filename):
    """Read boundary data from CSV and convert to coordinate list"""
    print(f"Reading {filename}")
    boundaries_df = pd.read_csv(f"{filename}")
    polygon_coords = extract_polygons(boundaries_df)
    return polygon_coords

def get_points_in_boundary(boundary_coords, plane_data_df):
    """Determine which aircraft points are inside boundary"""
    points_in_boundary = []
    
    for idx, row in plane_data_df.iterrows():
        if point_in_polygon(row['Longitude'], row['Latitude'], boundary_coords):
            points_in_boundary.append(idx)
    
    return points_in_boundary

def calculate_fuel_for_flights(flight_data, holdmode="BADA", saveopt2 = "All", boundary_df = None, change_speed = False):
    """
    Read CSV file and calculate fuel burn for each flight
    
    Args:
        csv_file_path: Path to input CSV file
        holdmode: Holding mode for fuel calculation ("BADA", "BADA_Cruise", "NATS")
        output_file: Optional output CSV file path
        saveopt2: Boundary filtering option ("All", "FIR", "TMA", "New")
                 - "All": Include all flight data points
                 - "FIR": Only include points within FIR boundary
                 - "TMA": Only include points within TMA radius
                 - "New": Only include points within custom boundary
        change_speed: Boolean flag to recalculate time points based on speed and distance
                     If True, recalculates timestamps using speed/distance relationships
                 
    Returns:
        DataFrame with original data plus fuel calculations
    """
    
    # Read boundary files based on saveopt2
    # if saveopt2 == "FIR":
    #     boundary_coords = get_boundaries(firboundaries_file)
    # elif saveopt2 == "New" or saveopt2 == "Custom":
    #     boundary_coords = get_boundaries(customboundaries_file)
    # else:
    #     boundary_coords = None  # Not needed for "All" or "TMA"

    # All boundaries should be parsed in from frontend

    if saveopt2 == "All":
        boundary_coords = None
    else:
        if boundary_df is not None:
            boundary_coords = extract_polygons(boundary_df)
        else:
            boundary_coords = None

    # Check if all required columns are present
    missing_columns = [col for col in required_columns if col not in flight_data.columns]

    if missing_columns:
        raise ValueError(f"The following required columns are missing: {missing_columns}")

    # Handle time formatting (equivalent to the R time fixing code)
    if 'Time' in flight_data.columns:
        # Fix dates without time (length 10: "YYYY-MM-DD")
        mask = flight_data['Time'].str.len() == 10
        flight_data.loc[mask, 'Time'] = flight_data.loc[mask, 'Time'] + " 00:00:00"

        # Convert to datetime
        flight_data['time'] = pd.to_datetime(flight_data['Time'], format='mixed', dayfirst=True, errors='raise')
    
    # Get unique flights (equivalent to R's lookup table)
    if 'ReferenceTime' in flight_data.columns:
        lookup_table = flight_data[['Callsign', 'ReferenceTime']].drop_duplicates()
    else:
        lookup_table = flight_data[['Callsign']].drop_duplicates()
        lookup_table['ReferenceTime'] = None
    
    lookup_table = lookup_table.reset_index(drop=True)
    print(lookup_table.head())
    all_flight_data = []

    print(f"Processing {len(lookup_table)} flights...")

    # Process each flight individually
    for i, row in lookup_table.iterrows():
        callsign = row['Callsign']
        ref_time = row.get('ReferenceTime')

        print(f"\nProcessing flight {i+1}/{len(lookup_table)}: {callsign}")

        # Filter data for this specific flight
        if ref_time is not None:
            plane_data = flight_data[
                (flight_data['Callsign'] == callsign) & 
                (flight_data['ReferenceTime'] == ref_time)
            ].copy()
        else:
            plane_data = flight_data[flight_data['Callsign'] == callsign].copy()

        # Skip flights with too few data points
        if len(plane_data) < 3:
            print(f"  Skipping {callsign} - insufficient data points ({len(plane_data)})")
            continue
        
        # print(plane_data.head())
        
        ##### Calculate new times from edited speeds #####
        if change_speed == True:
            print("  Recalculating time points based on speed and distance...")
            # Get speeds and distances
            speeds = plane_data['Speed'].values
            distances = plane_data['Distance'].values
            
            # Keep the first time point as reference
            new_times = [plane_data['time'].iloc[0]]

            # Calculate cumulative times based on speed and distance
            for i in range(len(plane_data) - 1):
                # Calculate average speed between current and next point
                avg_speed = (speeds[i] + speeds[i+1]) / 2

                # Convert knots to m/s (divide by 1.94384)
                speed_ms = avg_speed / 1.94384

                # Calculate time to travel distance
                travel_time_seconds = distances[i+1] / speed_ms

                # Calculate next time point
                next_time = new_times[i] + timedelta(seconds=travel_time_seconds)
                new_times.append(next_time)
            
            
            # print(plane_data['time'])
            # print(new_times)
            
            # Update the dataframe with new times
            plane_data['time'] = new_times
            
            
        # Calculate fuel flow for this flight
        fuel_data = get_fuel_flow_multi(
            plane_data['Aircraft Type'].tolist(),
            plane_data['Speed'].tolist(), 
            plane_data['Altitude'].tolist(),
            plane_data['time'].tolist(),
            plane_data['Latitude'].tolist(),
            plane_data['Longitude'].tolist(),
            holdmode
        )
            
        # print(fuel_data)
        # Select relevant columns (equivalent to [,c(1:3,5,6)])
        fuel_points = fuel_data[['fuel_at_point', 'fuel_flow_over_time', 'fuel_cumulative', 'phase', 'config']].copy()
        
        # Rename columns to match R output
        fuel_points.columns = ['BADA_fuel_at_point', 'BADA_fuel_over_time', 'BADA_fuel_cumulative', 'Phase', 'Config']

        ### FILTERING ###
        # Apply boundary filtering based on saveopt2 (equivalent to switch statement)
        # return positional indices
        if saveopt2 == "All":
            pointsinboundary = list(range(len(plane_data)))
        # elif saveopt2 == "FIR":
        #     boundary_indices = get_points_in_boundary(boundary_coords, plane_data)
        #     pointsinboundary = [plane_data.index.get_loc(idx) for idx in boundary_indices]
        # elif saveopt2 == "TMA":
            # tma_indices = plane_data[plane_data['DistToWSSS'] < (TMARadius * 1852)].index.tolist()
            # pointsinboundary = [plane_data.index.get_loc(idx) for idx in tma_indices]
        elif saveopt2 in ["New", "Custom", "TMA", "FIR"]:
            boundary_indices = get_points_in_boundary(boundary_coords, plane_data)
            pointsinboundary = [plane_data.index.get_loc(idx) for idx in boundary_indices]
        else:
            pointsinboundary = list(range(len(plane_data)))  # Default to All       
        
        if saveopt2 != "All":
            print(f"  Before {saveopt2} filtering: {len(plane_data)} points")
            plane_data = plane_data.iloc[pointsinboundary].reset_index(drop=True)
            fuel_points = fuel_points.iloc[pointsinboundary].reset_index(drop=True)
            print(f"  After {saveopt2} filtering: {len(plane_data)} points remaining")
        
        # Reset cumulative fuel to start from 0 for this flight
        fuel_points['BADA_fuel_cumulative'] = fuel_points['BADA_fuel_cumulative'] - fuel_points['BADA_fuel_cumulative'].min()
        
        # Combine plane data with fuel data
        plane_data_with_fuel = pd.concat([plane_data.reset_index(drop=True), fuel_points.reset_index(drop=True)], axis=1)

        all_flight_data.append(plane_data_with_fuel)

        print(f"  Completed {callsign} - {len(plane_data)} data points")

    if not all_flight_data:
        print("No flights were successfully processed!")
        return pd.DataFrame()

    # Combine all flight data
    final_data = pd.concat(all_flight_data, ignore_index=True)
    
    # Remove 'time' column
    if 'time' in final_data.columns:
        final_data['Time'] = final_data['time']
        final_data = final_data.drop(columns=['time'])

    print(f"Successfully processed {len(all_flight_data)} flights with {len(final_data)} total data points")

    # Save to file if specified

    return final_data

if __name__ == "__main__":
    # Configuration
    input_filename = "select_data_2_processed_speed_change.csv"
    input_file = f"C:\\Users\\KX\\Documents\\Work\\ASI TOOLS\\Fuel Consumption Calculator Translation\\Data_processed\\{input_filename}"
    holdmode = ["BADA", "BADA_Cruise", "NATS"][1]
    saveopt2 = "All"  # Can be "All", "FIR", "TMA", "New"
    # Read input data
    flight_data = pd.read_csv(input_file)
    # Run calculation
    result = calculate_fuel_for_flights(flight_data, holdmode, saveopt2, change_speed=True)
    print(result.head())

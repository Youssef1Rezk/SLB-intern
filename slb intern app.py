import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import numpy as np
from datetime import datetime
import json
import pickle
from wellarchitecturedesign import Tubular, Cement, Well, Tubing, Packer
import os
from scipy.optimize import fsolve
from scipy.interpolate import interp1d


# --- Session State Initialization ---
if "show_well_design" not in st.session_state:
    st.session_state.show_well_design = False
if "show_fluid_manager" not in st.session_state:
    st.session_state.show_fluid_manager = False
if "selected_tool" not in st.session_state:
    st.session_state.selected_tool = None
if 'fluids' not in st.session_state:
    st.session_state.fluids = {}
if 'selected_fluid' not in st.session_state:
    st.session_state.selected_fluid = None
if 'new_fluid_mode' not in st.session_state:
    st.session_state.new_fluid_mode = False
if 'survey_data_saved' not in st.session_state:
    st.session_state.survey_data_saved = {}
if 'current_survey_type' not in st.session_state:
    st.session_state.current_survey_type = ""
if "survey_df" not in st.session_state:
    st.session_state.survey_df = pd.DataFrame()
if "show_well_design" not in st.session_state:
    st.session_state.show_well_design = False
if "show_fluid_manager" not in st.session_state:
    st.session_state.show_fluid_manager = False
if "show_nodal_analysis" not in st.session_state:
    st.session_state.show_nodal_analysis = False
if "selected_tool" not in st.session_state:
    st.session_state.selected_tool = None
if "MD_heat" not in st.session_state:
    st.session_state.MD_heat = pd.DataFrame(columns=['MD(ft)', 'Ambient Temperature'])

if "TVD_heat" not in st.session_state:
    st.session_state.TVD_heat = pd.DataFrame(columns=['TVD(ft)', 'Ambient Temperature'])

if "Tubing" not in st.session_state:
    st.session_state.Tubing = pd.DataFrame(columns=[
        'Name','To MD','ID(in)','OD(in)','Wall thickness(in)','Roughness(in)'
    ])
if "casing_liners" not in st.session_state:
    st.session_state.casing_liners = pd.DataFrame(columns=['Section type','Name','From MD','To MD','ID(in)','OD(in)','Wall thickness(in)','Roughness(in)'])

if "Completions" not in st.session_state:
    st.session_state.Completions = pd.DataFrame(columns=[
        'Name','Geometry Profile','Fluid entry','Middle MD (ft)','Type','Active','IPR model'
    ])


# --- Sidebar ---
# --- Sidebar ---
with st.sidebar:
    # Main Page / Home Button (always visible at top)
    if st.button("🏠 Main Page", type="primary", use_container_width=True):
        # Reset all states to go back to main page
        st.session_state.show_well_design = False
        st.session_state.show_fluid_manager = False
        st.session_state.show_nodal_analysis = False
        st.session_state.selected_tool = None
        st.session_state.selected_fluid = None
        st.session_state.selected_completion = None
        st.session_state.new_fluid_mode = False
        st.session_state.new_completion_mode = False
        st.rerun()
    
    st.divider()
    
    # Well Design Section
    if not st.session_state.show_well_design:
        if st.button("🛢️ Well Design"):
            st.session_state.show_well_design = True
            st.session_state.show_fluid_manager = False
            st.session_state.show_nodal_analysis = False
    else:
        st.subheader("Well Design Options")
        st.session_state.selected_tool = st.selectbox(
            "Select a tool:",
            ["General", "Deviation survey", "Heat transfer", "Tubulars", 'Completions', 'Well schematics'],
            index=(["General", "Deviation survey", "Heat transfer", "Tubulars", 'Completions', 'Well schematics'].index(st.session_state.selected_tool)
                  if st.session_state.selected_tool else 0)
        )
        st.write(f"Selected: **{st.session_state.selected_tool}**")
        if st.button("❌ Close Well Design"):
            st.session_state.show_well_design = False
            st.session_state.selected_tool = None
    
    # Add some space between buttons
    st.write("")
    
    # Fluid Manager Section
    if not st.session_state.show_fluid_manager:
        if st.button("💧 Fluid Manager"):
            st.session_state.show_fluid_manager = True
            st.session_state.show_well_design = False
            st.session_state.show_nodal_analysis = False
    else:
        st.subheader("Fluid Manager")
        if st.button("❌ Close Fluid Manager"):
            st.session_state.show_fluid_manager = False
    
    # Add some space between buttons
    st.write("")
    
    # Nodal Analysis Section
    if not st.session_state.show_nodal_analysis:
        if st.button("📊 Nodal Analysis"):
            st.session_state.show_nodal_analysis = True
            st.session_state.show_well_design = False
            st.session_state.show_fluid_manager = False
    else:
        st.subheader("Nodal Analysis")
        if st.button("❌ Close Nodal Analysis"):
            st.session_state.show_nodal_analysis = False

# Fluid Manager
if st.session_state.show_fluid_manager:
    st.subheader('Fluid Manager 💧')
    
    if not st.session_state.selected_fluid:
        if not st.session_state.new_fluid_mode:
            # Show the add button
            if st.button("➕ Add New Fluid", type='primary'):
                st.session_state.new_fluid_mode = True
                st.rerun()  # Force immediate rerun
        else:
            # Show the input form
            name = st.text_input('Fluid name', placeholder='Enter fluid name')
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Save Fluid", type='primary'):
                    if name.strip():
                        if name not in st.session_state.fluids:
                            st.session_state.fluids[name] = {
                                'Created_date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                                'properties': {},
                                'notes': ''
                            }
                            st.success(f"✅ Fluid '{name}' added successfully!")
                            st.session_state.new_fluid_mode = False
                            st.rerun()
                        else:
                            st.error("❌ Fluid name already exists!")
                    else:
                        st.error("❌ Please enter a valid fluid name!")
            
            with col2:
                if st.button("❌ Cancel"):
                    st.session_state.new_fluid_mode = False
                    st.rerun()
    
    # Display fluids table
    if st.session_state.fluids:
        st.write('### Your Fluids')
        fluids_list = []
        
        for fluid_name, fluid_info in st.session_state.fluids.items():
            fluids_list.append({
                'Fluid_name': fluid_name,
                'Created_date': fluid_info.get('Created_date', 'N/A'),
                'Properties_count': len(fluid_info.get('properties', {}))
            })

        df_fluids = pd.DataFrame(fluids_list)
        
        # Create clickable buttons for each fluid
        st.write("Click on a fluid name to edit its properties:")
        
        for index, row in df_fluids.iterrows():
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                if st.button(f"🧪 {row['Fluid_name']}", key=f"fluid_{index}"):
                    st.session_state.selected_fluid = row['Fluid_name']
                    st.rerun()
            
            with col2:
                st.write(f"Created: {row['Created_date']}")
            
            with col3:
                st.write(f"Props: {row['Properties_count']}")
    
    # Fluid Properties Editor
    if st.session_state.selected_fluid:
        st.subheader(f"Editing: {st.session_state.selected_fluid} 🔧")
        
        # Get current fluid data and ensure properties key exists
        current_fluid = st.session_state.fluids[st.session_state.selected_fluid]
        
        # Initialize properties if it doesn't exist
        if 'properties' not in current_fluid:
            current_fluid['properties'] = {}
        
        # Initialize notes if it doesn't exist
        if 'notes' not in current_fluid:
            current_fluid['notes'] = ''
        
        # Properties Editor
        with st.form(key='fluid_properties_form'):
            st.write("### Fluid Properties")
            
            # Common fluid properties
            col1, col2 = st.columns(2)
            
            with col1:
                water_cut = st.number_input(
                    "Water cut",
                    value=current_fluid['properties'].get('water_cut', 0.0),  # Fixed key to match save key
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    help="Water cut in fraction"
                )
                
                GOR = st.number_input(
                    "GOR (SCF/STB)",
                    value=current_fluid['properties'].get('GOR', 0.0),
                    min_value=0.0,
                    step=0.001,
                    format="%.3f",
                    help="GOR in (SCF/STB)"
                )
                gas_specific_gravity = st.number_input(  # Fixed variable name consistency
                    "gas Specific Gravity",
                    value=current_fluid['properties'].get('gas_specific_gravity', 1.0),  # Fixed key to match save key
                    min_value=0.0,
                    step=0.001,
                    format="%.3f",
                    help="gas specific gravity (unitless)"
                )
                
            with col2:
                water_specific_gravity = st.number_input(  # Fixed variable name consistency
                    "Water Specific Gravity",
                    value=current_fluid['properties'].get('water_specific_gravity', 1.0),  # Fixed key to match save key
                    min_value=0.0,
                    step=0.001,
                    format="%.3f",
                    help="Water specific gravity (unitless)"
                )
                
                API = st.number_input(
                    "API",
                    value=current_fluid['properties'].get('API', 0.0),
                    step=0.1,
                    help="API gravity"
                )

            
            # Notes field
            st.write("### Notes")
            notes = st.text_area(
                "Additional Notes",
                value=current_fluid.get('notes', ''),
                height=100,
                help="Any additional information about this fluid"
            )
            
            # Form buttons
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                if st.form_submit_button("💾 Save Properties", type="primary"):
                    # Save all properties with consistent keys
                    current_fluid['properties'] = {
                        'water_cut': water_cut,  # Consistent key name
                        'GOR': GOR,
                        'water_specific_gravity': water_specific_gravity,  # Consistent key name
                        'gas_specific_gravity': gas_specific_gravity,
                        'API': API,
                    }
                    current_fluid['notes'] = notes
                    current_fluid['last_modified'] = datetime.now().strftime("%Y-%m-%d %H:%M")
                    st.success("✅ Properties saved successfully!")
                    st.rerun()
            
            with col2:
                if st.form_submit_button("← Back to Fluids List"):
                    st.session_state.selected_fluid = None
                    st.rerun()
            
            with col3:
                if st.form_submit_button("🗑️ Delete Fluid", type="secondary"):
                    # You might want to add a confirmation dialog here
                    if st.session_state.selected_fluid in st.session_state.fluids:
                        del st.session_state.fluids[st.session_state.selected_fluid]
                        st.session_state.selected_fluid = None
                        st.success("Fluid deleted!")
                        st.rerun()

     


#Well Design
if st.session_state.selected_tool:
    st.header(f"{st.session_state.selected_tool} ")

# General tool
if st.session_state.selected_tool == "General":
    st.text_input("Well Name")
    st.radio("Select the well type", ["production", "injection", "advanced"])
    st.radio("Check valve setting", ["Block none", "Block forward", "Block reverse", "Block both"])

# survey tool
if st.session_state.selected_tool == "Deviation survey":
    # Initialize session state variables if they don't exist
    if 'survey_type' not in st.session_state:
        st.session_state.survey_type = "Vertical"
    if 'depth_reference' not in st.session_state:
        st.session_state.depth_reference = "Original RKB"
    if 'wellhead_depth' not in st.session_state:
        st.session_state.wellhead_depth = 0.0
    if 'bottom_depth' not in st.session_state:
        st.session_state.bottom_depth = 0.0
    if 'current_survey_type' not in st.session_state:
        st.session_state.current_survey_type = "Vertical"
    if 'survey_data_saved' not in st.session_state:
        st.session_state.survey_data_saved = {}
    
    st.subheader("Survey Options")
    # Store survey type in session state
    survey_type = st.radio("Survey Type", ["Vertical", "2D", "3D"], 
                          index=["Vertical", "2D", "3D"].index(st.session_state.survey_type),
                          key="survey_type_radio")
    st.session_state.survey_type = survey_type

    st.subheader("Reference Option")
    # Store depth reference in session state
    depth_reference = st.radio("Depth Reference", ["Original RKB", "RKB", "GL", "MSL", "THF", "Wellhead"],
                              index=["Original RKB", "RKB", "GL", "MSL", "THF", "Wellhead"].index(st.session_state.depth_reference),
                              key="depth_reference_radio")
    st.session_state.depth_reference = depth_reference
    
    # Store wellhead depth in session state
    wellhead_depth = st.number_input("Wellhead Depth (ft)", min_value=0.0, 
                                    value=st.session_state.wellhead_depth,
                                    key="wellhead_depth_input")
    st.session_state.wellhead_depth = wellhead_depth
    
    # Store bottom depth in session state
    bottom_depth = st.number_input("Bottom Depth (ft)", min_value=0.0, 
                                  value=st.session_state.bottom_depth,
                                  key="bottom_depth_input")
    st.session_state.bottom_depth = bottom_depth

    # Only show table and plot for 2D/3D surveys
    if survey_type != "Vertical":
        # Define columns based on survey type
        columns_2d = ["MD (ft)", "TVD (ft)", "Horizontal Displacement (ft)", "Angle (°)"]
        columns_3d = ["MD (ft)", "TVD (ft)", "Horizontal Displacement (ft)", "Angle (°)", "Azimuth (°)", "Max Dogleg Severity (°/100ft)"]

        # Handle survey type change
        if st.session_state.current_survey_type != survey_type:
            # Load existing data or create new DataFrame
            if survey_type in st.session_state.survey_data_saved:
                st.session_state.survey_df = st.session_state.survey_data_saved[survey_type].copy()
            else:
                if survey_type == "2D":
                    st.session_state.survey_df = pd.DataFrame({col: [0.0] for col in columns_2d})
                elif survey_type == "3D":
                    st.session_state.survey_df = pd.DataFrame({col: [0.0] for col in columns_3d})
            
            # Update current survey type
            st.session_state.current_survey_type = survey_type

        # Ensure survey_df exists for current survey type
        if 'survey_df' not in st.session_state or st.session_state.survey_df.empty:
            if survey_type == "2D":
                st.session_state.survey_df = pd.DataFrame({col: [0.0] for col in columns_2d})
            elif survey_type == "3D":
                st.session_state.survey_df = pd.DataFrame({col: [0.0] for col in columns_3d})

        # Define column config for better editing experience
        if survey_type == "2D":
            column_config = {
                "MD (ft)": st.column_config.NumberColumn("MD (ft)", min_value=0.0, step=0.1),
                "TVD (ft)": st.column_config.NumberColumn("TVD (ft)", min_value=0.0, step=0.1),
                "Horizontal Displacement (ft)": st.column_config.NumberColumn("Horizontal Displacement (ft)", min_value=0.0, step=0.1),
                "Angle (°)": st.column_config.NumberColumn("Angle (°)", min_value=0.0, max_value=90.0, step=0.1),
            }
        else:  # 3D
            column_config = {
                "MD (ft)": st.column_config.NumberColumn("MD (ft)", min_value=0.0, step=0.1),
                "TVD (ft)": st.column_config.NumberColumn("TVD (ft)", min_value=0.0, step=0.1),
                "Horizontal Displacement (ft)": st.column_config.NumberColumn("Horizontal Displacement (ft)", min_value=0.0, step=0.1),
                "Angle (°)": st.column_config.NumberColumn("Angle (°)", min_value=0.0, max_value=90.0, step=0.1),
                "Azimuth (°)": st.column_config.NumberColumn("Azimuth (°)", min_value=0.0, max_value=360.0, step=0.1),
                "Max Dogleg Severity (°/100ft)": st.column_config.NumberColumn("Max Dogleg Severity (°/100ft)", min_value=0.0, step=0.1),
            }

        # Editable table
        st.subheader("Enter Survey Data")
        
        # Simple, static key
        editor_key = f"data_editor_{survey_type}"
        
        # Store the current data before showing the editor
        current_data = st.session_state.survey_df.copy()
        
        # Show the data editor
        edited_data = st.data_editor(
            current_data,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config=column_config,
            key=editor_key
        )
        
        # Always update session state with edited data
        st.session_state.survey_df = edited_data.copy()

        # Manual save button
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button(" Save Survey Data", key=f"save_{survey_type}", type="primary"):
                st.session_state.survey_data_saved[survey_type] = st.session_state.survey_df.copy()
                st.success(f" Survey data saved successfully!")
        
        with col2:
            # Load saved data button
            if survey_type in st.session_state.survey_data_saved:
                if st.button(" Load Saved Data", key=f"load_{survey_type}"):
                    st.session_state.survey_df = st.session_state.survey_data_saved[survey_type].copy()
                    st.success("Saved data loaded successfully!")
                    st.rerun()  # Only rerun when explicitly loading saved data

        # Show data status
        st.subheader("Data Status")
        col1, col2 = st.columns(2)
        
        with col1:
            if survey_type in st.session_state.survey_data_saved:
                rows_count = len(st.session_state.survey_data_saved[survey_type])
                st.success(f" Saved: {rows_count} rows")
            else:
                st.warning(" No saved data")
        
        with col2:
            current_rows = len(st.session_state.survey_df)
            st.info(f" Current: {current_rows} rows")
        
        # Clear data button (optional)
        st.subheader("Data Management")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(" Clear Current Data", key=f"clear_current_{survey_type}"):
                if survey_type == "2D":
                    st.session_state.survey_df = pd.DataFrame({col: [0.0] for col in columns_2d})
                elif survey_type == "3D":
                    st.session_state.survey_df = pd.DataFrame({col: [0.0] for col in columns_3d})
                st.warning("Current data cleared!")
                st.rerun()  # Only rerun when clearing data
        
        with col2:
            if survey_type in st.session_state.survey_data_saved:
                if st.button(" Delete Saved Data", key=f"delete_saved_{survey_type}"):
                    del st.session_state.survey_data_saved[survey_type]
                    st.warning(f" Saved {survey_type} data deleted!")
        
        # Visualization
        if not st.session_state.survey_df.empty and len(st.session_state.survey_df) > 0:
            # Check if we have actual data (not just zeros)
            try:
                numeric_cols = st.session_state.survey_df.select_dtypes(include=[np.number]).columns
                has_data = st.session_state.survey_df[numeric_cols].sum().sum() > 0
                
                if has_data:
                    st.subheader("Survey Visualization")
                    
                    # Create the plot
                    fig, ax = plt.subplots(figsize=(10, 8))
                    ax.plot(
                        st.session_state.survey_df["Horizontal Displacement (ft)"],
                        st.session_state.survey_df["TVD (ft)"],
                        marker="o",
                        linewidth=2,
                        markersize=6
                    )
                    ax.set_xlabel("Horizontal Displacement (ft)")
                    ax.set_ylabel("TVD (ft)")
                    ax.set_title(f"{survey_type} Survey: TVD vs Horizontal Displacement")
                    ax.grid(True, alpha=0.3)
                    ax.invert_yaxis()
                    st.pyplot(fig)
                    
                    # Show summary statistics
                    st.subheader("Survey Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Max TVD", f"{st.session_state.survey_df['TVD (ft)'].max():.1f} ft")
                    with col2:
                        st.metric("Max Horizontal", f"{st.session_state.survey_df['Horizontal Displacement (ft)'].max():.1f} ft")
                    with col3:
                        if "Angle (°)" in st.session_state.survey_df.columns:
                            st.metric("Max Angle", f"{st.session_state.survey_df['Angle (°)'].max():.1f}°")
                else:
                    st.info("Enter survey data in the table above to see visualization")
            except Exception as e:
                st.error(f"Error in visualization: {str(e)}")
                
    else:
        st.info("Vertical well selected - no deviation survey data needed")
        # Clear any existing survey data when vertical is selected
        if 'survey_df' in st.session_state:
            st.session_state.survey_df = pd.DataFrame()
        st.session_state.current_survey_type = "Vertical"
    
# Heat transfer
if st.session_state.selected_tool == "Heat transfer":
    st.subheader("Heat Transfer Parameters")
    Heat_transfer_coefficient = st.radio("Heat transfer coefficient", ["specify", "calculate"])
    if Heat_transfer_coefficient == "specify":
        U_value_input = st.radio("U value input", ["Single", "Multiple"])
        if U_value_input == "Single":
            average_U_value = st.number_input('Average U value')
            ambient_temperatue_input = st.radio("Ambient Temperature Value", ["Single", "Multiple"])
            if ambient_temperatue_input == "Single":
                soil_temp_wellhead = st.number_input('Soil temperature input (degF)')
            else:  # Multiple ambient temperatures
                depth_option = st.radio('Depth option', ["MD", "TVD"])
                if depth_option == "MD":
                    if st.session_state.MD_heat.empty or 'MD(ft)' not in st.session_state.MD_heat.columns:
                        st.session_state.MD_heat = pd.DataFrame(columns=['MD(ft)', 'Ambient Temperature'])

                    with st.form("Md_heat_form"):  # Changed form key to be unique
                        edited_Md_heat_df = st.data_editor(
                            st.session_state.MD_heat,
                            column_config={
                                "MD(ft)": st.column_config.NumberColumn("MD(ft)", required=True, min_value=0.0),
                                'Ambient Temperature': st.column_config.NumberColumn('Ambient Temperature', required=True, min_value=0.0),
                            },
                            hide_index=True,
                            num_rows='dynamic',
                            key='edited_Md_heat',
                            disabled=False
                        )

                        col1, col2 = st.columns(2)
                        with col1:
                            submitted = st.form_submit_button(" Save Data")
                        if submitted:
                            if not edited_Md_heat_df.empty and not edited_Md_heat_df.isnull().values.any():
                                st.session_state.MD_heat = edited_Md_heat_df
                                st.session_state.edit_complete = True
                                st.success("Data saved successfully!")
                            else:
                                st.warning("Please fill in all data before saving.")

                    if not st.session_state.MD_heat.empty and 'MD(ft)' in st.session_state.MD_heat.columns and 'Ambient Temperature' in st.session_state.MD_heat.columns:
                        try:
                            fig, ax = plt.subplots()
                            ax.plot(
                                st.session_state.MD_heat['Ambient Temperature'],
                                st.session_state.MD_heat['MD(ft)'],
                                marker='o'
                            )
                            ax.set_xlabel("Ambient Temperature")
                            ax.set_ylabel("MD (ft)")
                            ax.set_title("MD vs. Ambient Temperature")
                            ax.grid(True)
                            ax.invert_yaxis()
                            st.pyplot(fig)
                        except KeyError:
                            st.warning("Data columns are missing. Please re-enter your data.")
                    else:
                        st.info("Please enter and save data to view the plot.")
                else: # depth_option == "TVD"
                    if st.session_state.TVD_heat.empty or 'TVD(ft)' not in st.session_state.TVD_heat.columns:
                        st.session_state.TVD_heat = pd.DataFrame(columns=['TVD(ft)', 'Ambient Temperature'])

                    with st.form("TVD_heat_form"): # Changed form key to be unique
                        edited_TVD_heat_df = st.data_editor(
                            st.session_state.TVD_heat,
                            column_config={
                                "TVD(ft)": st.column_config.NumberColumn("TVD(ft)", required=True, min_value=0.0),
                                'Ambient Temperature': st.column_config.NumberColumn('Ambient Temperature', required=True, min_value=0.0),
                            },
                            hide_index=True,
                            num_rows='dynamic',
                            key='edited_TVD_heat_data', # Changed data_editor key to be unique
                            disabled=False
                        )

                        col1, col2 = st.columns(2)
                        with col1:
                            submitted = st.form_submit_button(" Save Data")
                        if submitted:
                            if not edited_TVD_heat_df.empty and not edited_TVD_heat_df.isnull().values.any():
                                st.session_state.TVD_heat = edited_TVD_heat_df 
                                st.session_state.edit_complete = True
                                st.success("Data saved successfully!")
                            else:
                                st.warning("Please fill in all data before saving.")

                    if not st.session_state.TVD_heat.empty and 'TVD(ft)' in st.session_state.TVD_heat.columns and 'Ambient Temperature' in st.session_state.TVD_heat.columns:
                        try:
                            fig, ax = plt.subplots()
                            ax.plot(
                                st.session_state.TVD_heat['Ambient Temperature'],
                                st.session_state.TVD_heat['TVD(ft)'],
                                marker='o'
                            )
                            ax.set_xlabel("Ambient Temperature")
                            ax.set_ylabel("TVD (ft)")
                            ax.set_title("TVD vs. Ambient Temperature")
                            ax.grid(True)
                            ax.invert_yaxis()
                            st.pyplot(fig)
                        except KeyError:
                            st.warning("Data columns are missing. Please re-enter your data.")
                    else:
                        st.info("Please enter and save data to view the plot.")
        else:
             ambient_temperatue_input = st.radio("Ambient Temperature Value", ["Single", "Multiple"])
             if ambient_temperatue_input == "Single":
                  soil_temp_wellhead = st.number_input('Soil temperature input (degF)')
                  depth_option = st.radio('Depth option', ["MD", "TVD"])
                  if depth_option == "MD":
                    if st.session_state.MD_heat.empty or 'MD(ft)' not in st.session_state.MD_heat.columns:
                        st.session_state.MD_heat = pd.DataFrame(columns=['MD(ft)', 'U value'])

                    with st.form("Md_heat_form"):  # Changed form key to be unique
                        edited_Md_heat_df = st.data_editor(
                            st.session_state.MD_heat,
                            column_config={
                                "MD(ft)": st.column_config.NumberColumn("MD(ft)", required=True, min_value=0.0),
                                'U value': st.column_config.NumberColumn('U value', required=True, min_value=0.0),
                            },
                            hide_index=True,
                            num_rows='dynamic',
                            key='edited_Md_heat',
                            disabled=False
                        )

                        col1, col2 = st.columns(2)
                        with col1:
                            submitted = st.form_submit_button(" Save Data")
                        if submitted:
                            if not edited_Md_heat_df.empty and not edited_Md_heat_df.isnull().values.any():
                                st.session_state.MD_heat = edited_Md_heat_df 
                                st.session_state.edit_complete = True
                                st.success("Data saved successfully!")
                            else:
                                st.warning("Please fill in all data before saving.")

                    if not st.session_state.MD_heat.empty and 'MD(ft)' in st.session_state.MD_heat.columns and 'U value' in st.session_state.MD_heat.columns:
                        try:
                            fig, ax = plt.subplots()
                            ax.plot(
                                st.session_state.MD_heat['U value'],
                                st.session_state.MD_heat['MD(ft)'],
                                marker='o'
                            )
                            ax.set_xlabel("U value")
                            ax.set_ylabel("MD (ft)")
                            ax.set_title("MD vs.U value")
                            ax.grid(True)
                            ax.invert_yaxis()
                            st.pyplot(fig)
                        except KeyError:
                            st.warning("Data columns are missing. Please re-enter your data.")
                    else:
                        st.info("Please enter and save data to view the plot.")
                  else: # depth_option == "TVD"
                    if st.session_state.TVD_heat.empty or 'TVD(ft)' not in st.session_state.TVD_heat.columns:
                        st.session_state.TVD_heat = pd.DataFrame(columns=['TVD(ft)', 'U value'])

                    with st.form("TVD_heat_form"): # Changed form key to be unique
                        edited_TVD_heat_df = st.data_editor(
                            st.session_state.TVD_heat,
                            column_config={
                                "TVD(ft)": st.column_config.NumberColumn("TVD(ft)", required=True, min_value=0.0),
                                'U value': st.column_config.NumberColumn('U value', required=True, min_value=0.0),
                            },
                            hide_index=True,
                            num_rows='dynamic',
                            key='edited_TVD_heat_data', # Changed data_editor key to be unique
                            disabled=False
                        )

                        col1, col2 = st.columns(2)
                        with col1:
                            submitted = st.form_submit_button(" Save Data")
                        if submitted:
                            if not edited_TVD_heat_df.empty and not edited_TVD_heat_df.isnull().values.any():
                                st.session_state.TVD_heat = edited_TVD_heat_df 
                                st.session_state.edit_complete = True
                                st.success("Data saved successfully!")
                            else:
                                st.warning("Please fill in all data before saving.")

                    if not st.session_state.TVD_heat.empty and 'TVD(ft)' in st.session_state.TVD_heat.columns and 'U value' in st.session_state.TVD_heat.columns:
                        try:
                            fig, ax = plt.subplots()
                            ax.plot(
                                st.session_state.TVD_heat['U value'],
                                st.session_state.TVD_heat['TVD(ft)'],
                                marker='o'
                            )
                            ax.set_xlabel("U value")
                            ax.set_ylabel("TVD (ft)")
                            ax.set_title("TVD vs. U value")
                            ax.grid(True)
                            ax.invert_yaxis()
                            st.pyplot(fig)
                        except KeyError:
                            st.warning("Data columns are missing. Please re-enter your data.")
                    else:
                        st.info("Please enter and save data to view the plot.")
             else :
                  depth_option = st.radio('Depth option', ["MD", "TVD"])
                  if depth_option == "MD":
                    if st.session_state.MD_heat.empty or 'MD(ft)' not in st.session_state.MD_heat.columns:
                        st.session_state.MD_heat = pd.DataFrame(columns=['MD(ft)', 'U value','Ambient Temperature'])

                    with st.form("Md_heat_form"):  # Changed form key to be unique
                        edited_Md_heat_df = st.data_editor(
                            st.session_state.MD_heat,
                            column_config={
                                "MD(ft)": st.column_config.NumberColumn("MD(ft)", required=True, min_value=0.0),
                                'U value': st.column_config.NumberColumn('U value', required=True, min_value=0.0),
                                "Ambient Temperature": st.column_config.NumberColumn("Ambient Temperature", required=True, min_value=0.0)
                            },
                            hide_index=True,
                            num_rows='dynamic',
                            key='edited_Md_heat',
                            disabled=False
                        )

                        col1, col2 = st.columns(2)
                        with col1:
                            submitted = st.form_submit_button(" Save Data")
                        if submitted:
                            if not edited_Md_heat_df.empty and not edited_Md_heat_df.isnull().values.any():
                                st.session_state.MD_heat = edited_Md_heat_df 
                                st.session_state.edit_complete = True
                                st.success("Data saved successfully!")
                            else:
                                st.warning("Please fill in all data before saving.")
                    if not st.session_state.MD_heat.empty and 'MD(ft)' in st.session_state.MD_heat.columns and 'U value' in st.session_state.MD_heat.columns and 'Ambient Temperature' in st.session_state.MD_heat.columns :
                        select_X_axis= st.radio("Select bottom X axis", ["U value", "Ambient Temperature"])
                        if select_X_axis=="U value":
                            try:
                                fig, ax = plt.subplots()
                                ax.plot(
                                    st.session_state.MD_heat['U value'],
                                    st.session_state.MD_heat['MD(ft)'],
                                    marker='o'
                                )
                                ax.set_xlabel("U value")
                                ax.set_ylabel("MD (ft)")
                                ax.set_title("MD vs.U value")
                                ax.grid(True)
                                ax.invert_yaxis()
                                st.pyplot(fig)
                            except KeyError:
                                st.warning("Data columns are missing. Please re-enter your data.")
                        else:
                            try:
                                fig, ax = plt.subplots()
                                ax.plot(
                                    st.session_state.MD_heat['Ambient Temperature'],
                                    st.session_state.MD_heat['MD(ft)'],
                                    marker='o'
                                )
                                ax.set_xlabel("Ambient Temperature")
                                ax.set_ylabel("MD (ft)")
                                ax.set_title("MD vs.Ambient Temperature")
                                ax.grid(True)
                                ax.invert_yaxis()
                                st.pyplot(fig)
                            except KeyError:
                                st.warning("Data columns are missing. Please re-enter your data.")
                    else:
                        st.info("Please enter and save data to view the plot.")
                  else: # depth_option == "TVD"
                    if st.session_state.TVD_heat.empty or 'TVD(ft)' not in st.session_state.TVD_heat.columns:
                        st.session_state.TVD_heat = pd.DataFrame(columns=['TVD(ft)', 'U value','Ambient Temperature'])

                    with st.form("TVD_heat_form"): # Changed form key to be unique
                        edited_TVD_heat_df = st.data_editor(
                            st.session_state.TVD_heat,
                            column_config={
                                "TVD(ft)": st.column_config.NumberColumn("TVD(ft)", required=True, min_value=0.0),
                                'U value': st.column_config.NumberColumn('U value', required=True, min_value=0.0),
                                "Ambient Temperature": st.column_config.NumberColumn("Ambient Temperature", required=True, min_value=0.0),
                            },
                            hide_index=True,
                            num_rows='dynamic',
                            key='edited_TVD_heat_data', # Changed data_editor key to be unique
                            disabled=False
                        )

                        col1, col2 = st.columns(2)
                        with col1:
                            submitted = st.form_submit_button(" Save Data")
                        if submitted:
                            if not edited_TVD_heat_df.empty and not edited_TVD_heat_df.isnull().values.any():
                                st.session_state.TVD_heat = edited_TVD_heat_df 
                                st.session_state.edit_complete = True
                                st.success("Data saved successfully!")
                            else:
                                st.warning("Please fill in all data before saving.")
                    if not st.session_state.TVD_heat.empty and 'TVD(ft)' in st.session_state.TVD_heat.columns and 'U value' in st.session_state.TVD_heat.columns and 'Ambient Temperature' in st.session_state.TVD_heat.columns :
                        select_X_axis= st.radio("Select bottom X axis", ["U value", "Ambient Temperature"])
                        if select_X_axis=="U value":
                            try:
                                fig, ax = plt.subplots()
                                ax.plot(
                                    st.session_state.TVD_heat['U value'],
                                    st.session_state.TVD_heat['TVD(ft)'],
                                    marker='o'
                                )
                                ax.set_xlabel("U value")
                                ax.set_ylabel("TVD (ft)")
                                ax.set_title("TVD vs. U value")
                                ax.grid(True)
                                ax.invert_yaxis()
                                st.pyplot(fig)
                            except KeyError:
                                st.warning("Data columns are missing. Please re-enter your data.")
                        else:
                            try:
                                fig, ax = plt.subplots()
                                ax.plot(
                                    st.session_state.TVD_heat['Ambient Temperature'],
                                    st.session_state.TVD_heat['TVD(ft)'],
                                    marker='o'
                                )
                                ax.set_xlabel("Ambient Temperature")
                                ax.set_ylabel("TVD (ft)")
                                ax.set_title("TVD vs. Ambient Temperature")
                                ax.grid(True)
                                ax.invert_yaxis()
                                st.pyplot(fig)
                            except KeyError:
                                st.warning("Data columns are missing. Please re-enter your data.")
                    else:
                        st.info("Please enter and save data to view the plot.")
    else:
        production_injection_time = st.number_input('Production/injection time (hr)')
        ambient_temperatue_input = st.radio("Ambient Temperature Value", ["Single", "Multiple"])
        if ambient_temperatue_input=="Single":
            soil_temp_wellhead = st.number_input('Soil temperature input (degF)')
            depth_option = st.radio('Depth option', ["MD", "TVD"])
            if depth_option == "MD":
                if st.session_state.MD_heat.empty or 'MD(ft)' not in st.session_state.MD_heat.columns:
                        st.session_state.MD_heat = pd.DataFrame(columns=['MD(ft)','Ground denisty','Ground K','Ground Cp'])
                with st.form("Md_heat_form"):  # Changed form key to be unique
                        edited_Md_heat_df = st.data_editor(
                            st.session_state.MD_heat,
                            column_config={
                                "MD(ft)": st.column_config.NumberColumn("MD(ft)", required=True, min_value=0.0),
                                'Ground denisty': st.column_config.NumberColumn('Ground denisty', required=True, min_value=0.0),
                                "Ground K": st.column_config.NumberColumn("Ground K", required=True, min_value=0.0),
                                "Ground Cp": st.column_config.NumberColumn("Ground Cp", required=True, min_value=0.0),
                            },
                            hide_index=True,
                            num_rows='dynamic',
                            key='edited_Md_heat',
                            disabled=False
                        )

                        col1, col2 = st.columns(2)
                        with col1:
                            submitted = st.form_submit_button(" Save Data")
                        if submitted:
                            if not edited_Md_heat_df.empty and not edited_Md_heat_df.isnull().values.any():
                                st.session_state.MD_heat = edited_Md_heat_df 
                                st.session_state.edit_complete = True
                                st.success("Data saved successfully!")
                            else:
                                st.warning("Please fill in all data before saving.")
                required_columns = ['MD(ft)','Ground denisty','Ground K','Ground Cp']
                if (
    not st.session_state.MD_heat.empty
    and all(col in st.session_state.MD_heat.columns for col in required_columns)
):
                    select_X_axis = st.radio(
        "Select bottom X axis",
        ['Ground denisty','Ground K','Ground Cp']
    )
                    x_axis_map = {
            'Ground denisty': ('Ground denisty', 'Ground denisty'),
            'Ground K': ('Ground K', 'Ground K'),
            'Ground Cp': ('Ground Cp','Ground Cp')
        }
                    col_name, label = x_axis_map[select_X_axis]
                    try:
                        fig, ax = plt.subplots()
                        ax.plot(
                            st.session_state.MD_heat[col_name],
                            st.session_state.MD_heat['MD(ft)'],
                            marker='o'
                        )
                        ax.set_xlabel(label)
                        ax.set_ylabel("MD (ft)")
                        ax.set_title(f"MD vs. {label}")
                        ax.grid(True)
                        ax.invert_yaxis()
                        st.pyplot(fig)
                    except KeyError:
                        st.warning("Data columns are missing. Please re-enter your data.")
            else:
                if st.session_state.TVD_heat.empty or 'TVD(ft)' not in st.session_state.TVD_heat.columns:
                        st.session_state.TVD_heat = pd.DataFrame(columns=['TVD(ft)','Ground denisty','Ground K','Ground Cp'])
                with st.form("TVD_heat_form"): # Changed form key to be unique
                        edited_TVD_heat_df = st.data_editor(
                            st.session_state.TVD_heat,
                            column_config={
                                "TVD(ft)": st.column_config.NumberColumn("TVD(ft)", required=True, min_value=0.0),
                                'Ground denisty': st.column_config.NumberColumn('Ground denisty', required=True, min_value=0.0),
                                "Ground K": st.column_config.NumberColumn("Ground K", required=True, min_value=0.0),
                                "Ground Cp": st.column_config.NumberColumn("Ground Cp", required=True, min_value=0.0),
                            },
                            hide_index=True,
                            num_rows='dynamic',
                            key='edited_TVD_heat',
                            disabled=False
                        )

                        col1, col2 = st.columns(2)
                        with col1:
                            submitted = st.form_submit_button(" Save Data")
                        if submitted:
                            if not edited_TVD_heat_df.empty and not edited_TVD_heat_df.isnull().values.any():
                                st.session_state.TVD_heat = edited_TVD_heat_df 
                                st.session_state.edit_complete = True
                                st.success("Data saved successfully!")
                            else:
                                st.warning("Please fill in all data before saving.")
                required_columns = ['TVD(ft)','Ground denisty','Ground K','Ground Cp']
                if (
    not st.session_state.TVD_heat.empty
    and all(col in st.session_state.TVD_heat.columns for col in required_columns)
):
                    select_X_axis = st.radio(
        "Select bottom X axis",
        ['Ground denisty','Ground K','Ground Cp']
    )
                    x_axis_map = {
            'Ground denisty': ('Ground denisty', 'Ground denisty'),
            'Ground K': ('Ground K', 'Ground K'),
            'Ground Cp': ('Ground Cp','Ground Cp')
        }
                    col_name, label = x_axis_map[select_X_axis]
                    try:
                        fig, ax = plt.subplots()
                        ax.plot(
                            st.session_state.TVD_heat[col_name],
                            st.session_state.TVD_heat['TVD(ft)'],
                            marker='o'
                        )
                        ax.set_xlabel(label)
                        ax.set_ylabel("TVD (ft)")
                        ax.set_title(f"TVD vs. {label}")
                        ax.grid(True)
                        ax.invert_yaxis()
                        st.pyplot(fig)
                    except KeyError:
                        st.warning("Data columns are missing. Please re-enter your data.")

        else :
            depth_option = st.radio('Depth option', ["MD", "TVD"])
            if depth_option == "MD":
                if st.session_state.MD_heat.empty or 'MD(ft)' not in st.session_state.MD_heat.columns:
                        st.session_state.MD_heat = pd.DataFrame(columns=['MD(ft)','Ground denisty','Ground K','Ground Cp','Ambient Temperature'])
                with st.form("Md_heat_form"):  # Changed form key to be unique
                        edited_Md_heat_df = st.data_editor(
                            st.session_state.MD_heat,
                            column_config={
                                "MD(ft)": st.column_config.NumberColumn("MD(ft)", required=True, min_value=0.0),
                                'Ground denisty': st.column_config.NumberColumn('Ground denisty', required=True, min_value=0.0),
                                "Ground K": st.column_config.NumberColumn("Ground K", required=True, min_value=0.0),
                                "Ground Cp": st.column_config.NumberColumn("Ground Cp", required=True, min_value=0.0),
                                'Ambient Temperature': st.column_config.NumberColumn('Ambient Temperature', required=True, min_value=0.0),
                            },
                            hide_index=True,
                            num_rows='dynamic',
                            key='edited_Md_heat',
                            disabled=False
                        )

                        col1, col2 = st.columns(2)
                        with col1:
                            submitted = st.form_submit_button(" Save Data")
                        if submitted:
                            if not edited_Md_heat_df.empty and not edited_Md_heat_df.isnull().values.any():
                                st.session_state.MD_heat = edited_Md_heat_df 
                                st.session_state.edit_complete = True
                                st.success("Data saved successfully!")
                            else:
                                st.warning("Please fill in all data before saving.")
                required_columns = ['MD(ft)','Ground denisty','Ground K','Ground Cp','Ambient Temperature']
                if (
    not st.session_state.MD_heat.empty
    and all(col in st.session_state.MD_heat.columns for col in required_columns)
):
                    select_X_axis = st.radio(
        "Select bottom X axis",
        ['Ground denisty','Ground K','Ground Cp','Ambient Temperature']
    )
                    x_axis_map = {
            'Ground denisty': ('Ground denisty', 'Ground denisty'),
            'Ground K': ('Ground K', 'Ground K'),
            'Ground Cp': ('Ground Cp','Ground Cp'),
            'Ambient Temperature': ('Ambient Temperature','Ambient Temperature')
        }
                    col_name, label = x_axis_map[select_X_axis]
                    try:
                        fig, ax = plt.subplots()
                        ax.plot(
                            st.session_state.MD_heat[col_name],
                            st.session_state.MD_heat['MD(ft)'],
                            marker='o'
                        )
                        ax.set_xlabel(label)
                        ax.set_ylabel("MD (ft)")
                        ax.set_title(f"MD vs. {label}")
                        ax.grid(True)
                        ax.invert_yaxis()
                        st.pyplot(fig)
                    except KeyError:
                        st.warning("Data columns are missing. Please re-enter your data.")
            else:
                if st.session_state.TVD_heat.empty or 'TVD(ft)' not in st.session_state.TVD_heat.columns:
                        st.session_state.TVD_heat = pd.DataFrame(columns=['TVD(ft)','Ground denisty','Ground K','Ground Cp','Ambient Temperature'])
                with st.form("TVD_heat_form"): # Changed form key to be unique
                        edited_TVD_heat_df = st.data_editor(
                            st.session_state.TVD_heat,
                            column_config={
                                "TVD(ft)": st.column_config.NumberColumn("TVD(ft)", required=True, min_value=0.0),
                                'Ground denisty': st.column_config.NumberColumn('Ground denisty', required=True, min_value=0.0),
                                "Ground K": st.column_config.NumberColumn("Ground K", required=True, min_value=0.0),
                                "Ground Cp": st.column_config.NumberColumn("Ground Cp", required=True, min_value=0.0),
                                'Ambient Temperature': st.column_config.NumberColumn('Ambient Temperature', required=True, min_value=0.0),
                            },
                            hide_index=True,
                            num_rows='dynamic',
                            key='edited_TVD_heat_data', # Changed data_editor key to be unique
                            disabled=False
                        )

                        col1, col2 = st.columns(2)
                        with col1:
                            submitted = st.form_submit_button(" Save Data")
                        if submitted:
                            if not edited_TVD_heat_df.empty and not edited_TVD_heat_df.isnull().values.any():
                                st.session_state.TVD_heat = edited_TVD_heat_df 
                                st.session_state.edit_complete = True
                                st.success("Data saved successfully!")
                            else:
                                st.warning("Please fill in all data before saving.")
                required_columns = ['TVD(ft)','Ground denisty','Ground K','Ground Cp','Ambient Temperature']
                if (
    not st.session_state.TVD_heat.empty
    and all(col in st.session_state.TVD_heat.columns for col in required_columns)
):
                    select_X_axis = st.radio(
        "Select bottom X axis",
        ['Ground denisty','Ground K','Ground Cp','Ambient Temperature']
    )
                    x_axis_map = {
            'Ground denisty': ('Ground denisty', 'Ground denisty'),
            'Ground K': ('Ground K', 'Ground K'),
            'Ground Cp': ('Ground Cp','Ground Cp'),
            'Ambient Temperature': ('Ambient Temperature','Ambient Temperature')
        }
                    col_name, label = x_axis_map[select_X_axis]
                    try:
                        fig, ax = plt.subplots()
                        ax.plot(
                            st.session_state.TVD_heat[col_name],
                            st.session_state.TVD_heat['TVD(ft)'],
                            marker='o'
                        )
                        ax.set_xlabel(label)
                        ax.set_ylabel("TVD (ft)")
                        ax.set_title(f"TVD vs. {label}")
                        ax.grid(True)
                        ax.invert_yaxis()
                        st.pyplot(fig)
                    except KeyError:
                        st.warning("Data columns are missing. Please re-enter your data.")
                        
# Tubulars
if 'selected_tool' not in st.session_state:
    st.session_state.selected_tool = "Tubulars"
if 'casing_liners' not in st.session_state:
    st.session_state.casing_liners = pd.DataFrame(columns=['Section type','Name','From MD','To MD','ID(in)','OD(in)','Wall thickness(in)','Roughness(in)'])
if 'Tubing' not in st.session_state:
    st.session_state.Tubing = pd.DataFrame(columns=['Name','To MD','ID(in)','OD(in)','Wall thickness(in)','Roughness(in)'])
if 'casing_edit_complete' not in st.session_state:
    st.session_state.casing_edit_complete = False
if 'tubing_edit_complete' not in st.session_state:
    st.session_state.tubing_edit_complete = False
if 'additional_data' not in st.session_state:
    st.session_state.additional_data = {}
if 'additional_data2' not in st.session_state:
    st.session_state.additional_data2 = {}


# The main application logic starts here, wrapped in a conditional block
if st.session_state.selected_tool == "Tubulars":
    
    st.title("Tubulars Data Entry")
    st.markdown("---")

    # ------------------------------------------------
    # Section 1: Casing and Liner Data Editor
    # ------------------------------------------------
    with st.form("casing_liner_main_form"):
        st.header('Casing/Liner')
        edited_casing_liners_df = st.data_editor(
            st.session_state.casing_liners,
            column_config={
                "Section type": st.column_config.SelectboxColumn("Section type",options=['Casing','Liner','Open hole'] ,required=True ),
                'Name': st.column_config.TextColumn('Name', required=True),
                "From MD": st.column_config.NumberColumn("From MD", required=True, min_value=0.0),
                "To MD": st.column_config.NumberColumn("To MD", required=True, min_value=0.0),
                'ID(in)': st.column_config.NumberColumn('ID(in)', required=True, min_value=0.0),
                'OD(in)': st.column_config.NumberColumn('OD(in)', required=True, min_value=0.0),
                'Wall thickness(in)': st.column_config.NumberColumn('Wall thickness(in)', required=True, min_value=0.0),
                'Roughness(in)': st.column_config.NumberColumn('Roughness(in)', required=True, min_value=0.00),
            },
            hide_index=True,
            num_rows='dynamic',
            key='edited_casing_liner_data_editor',
            disabled=False
        )
        
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("✅ Save Casing/Liner Data")

        if submitted:
            # Check if any row has incomplete data
            incomplete_rows = False
            for index, row in edited_casing_liners_df.iterrows():
                if row.isnull().any():
                    incomplete_rows = True
                    st.error(f"Please complete all fields in row {index+1} before saving.")
                    break
            
            if not incomplete_rows:
                st.session_state.casing_liners = edited_casing_liners_df
                st.session_state.casing_edit_complete = True
                st.success("Casing/Liner data saved successfully!")

    st.markdown("---")

    # The additional details section for Casing/Liner
    if st.session_state.casing_edit_complete:
        st.subheader('Additional Casing/Liner Details')
        with st.form("additional_details_form_casing"):
            for index, row in st.session_state.casing_liners.iterrows():
                section_type = row['Section type']
                section_name = row['Name']
                st.subheader(f"Properties for: {section_name} ({section_type})")

                with st.container():
                    if section_type in ['Casing', 'Liner']:
                        st.write(f'{section_type} Properties')
                        st.number_input(
                            'Density (ibm/ft3)',
                            min_value=0.0,
                            value=st.session_state.additional_data.get(f'density_{index}', 0.0),
                            key=f'density_input_{index}'
                        )
                        st.number_input(
                            'Thermal Conductivity (BTU(h.degF.ft))',
                            min_value=0.0,
                            value=st.session_state.additional_data.get(f'thermal_cond_{index}', 27.75),
                            key=f'thermal_cond_input_{index}'
                        )
                        st.number_input(
                            'Borehole Diameter (in))',
                            min_value=0.0,
                            value=st.session_state.additional_data.get(f'borehole_diam_{index}', 0.0),
                            key=f'borehole_diam_input_{index}'
                        )
                        st.write('Annulus Material')
                        st.number_input(
                            'Cement top (ft)',
                            min_value=0.0,
                            value=st.session_state.additional_data.get(f'cement_top_{index}', 0.0),
                            key=f'cement_top_input_{index}'
                        )
                        st.number_input(
                            'Cement Denisty (Ibm/Gal)',
                            min_value=0.0,
                            value=st.session_state.additional_data.get(f'cement_denisty_{index}', 15.85627),
                            key=f'cement_density_input_{index}'
                        )
                        st.number_input(
                            'Cement thermal conductivit (BTU(h.degF.ft)',
                            min_value=0.0,
                            value=st.session_state.additional_data.get(f'cement_thermal_cond_{index}', 0.9),
                            key=f'cement_thermal_cond_input_{index}'
                        )
                    elif section_type == 'Open hole':
                        st.number_input(
                            'Borehole diameter (in)', 
                            min_value=0.0, 
                            value=st.session_state.additional_data.get(f'OpenHole_wellbore_diameter_{index}', 0.0),
                            key=f'OpenHole_wellbore_diameter_{index}'
                        )

            submitted_details = st.form_submit_button("💾 Save Additional Details")

            if submitted_details:
                # Update the main storage dictionary with values from the form's keys
                for index in range(len(st.session_state.casing_liners)):
                    section_type = st.session_state.casing_liners.loc[index, 'Section type']
                    if section_type in ['Casing', 'Liner']:
                        st.session_state.additional_data[f'density_{index}'] = st.session_state[f'density_input_{index}']
                        st.session_state.additional_data[f'thermal_cond_{index}'] = st.session_state[f'thermal_cond_input_{index}']
                        st.session_state.additional_data[f'borehole_diam_{index}'] = st.session_state[f'borehole_diam_input_{index}']
                        st.session_state.additional_data[f'cement_top_{index}'] = st.session_state[f'cement_top_input_{index}']
                        st.session_state.additional_data[f'cement_denisty_{index}'] = st.session_state[f'cement_density_input_{index}']
                        st.session_state.additional_data[f'cement_thermal_cond_{index}'] = st.session_state[f'cement_thermal_cond_input_{index}']
                    elif section_type == 'Open hole':
                        st.session_state.additional_data[f'OpenHole_wellbore_diameter_{index}'] = st.session_state[f'OpenHole_wellbore_diameter_{index}']
                st.success("Additional details saved!")

    st.markdown("---")
    
    # ------------------------------------------------
    # Section 2: Tubing Data Editor
    # ------------------------------------------------
    if st.session_state.Tubing.empty or 'Name' not in st.session_state.Tubing.columns:
        st.session_state.Tubing = pd.DataFrame(columns=['Name','To MD','ID(in)','OD(in)','Wall thickness(in)','Roughness(in)'])

    with st.form("tubing_main_form"):
        st.header('Tubing')
        edited_Tubing_df = st.data_editor(
            st.session_state.Tubing,
            column_config={
                'Name': st.column_config.TextColumn('Name', required=True),
                "To MD": st.column_config.NumberColumn("To MD", required=True, min_value=0.0),
                'ID(in)': st.column_config.NumberColumn('ID(in)', required=True, min_value=0.0),
                'OD(in)': st.column_config.NumberColumn('OD(in)', required=True, min_value=0.0),
                'Wall thickness(in)': st.column_config.NumberColumn('Wall thickness(in)', required=True, min_value=0.0),
                'Roughness(in)': st.column_config.NumberColumn('Roughness(in)', required=True, min_value=0.0),
            },
            hide_index=True,
            num_rows='dynamic',
            key='edited_tubing_data_editor',
            disabled=False
        )
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("✅ Save Tubing Data")

        if submitted:
            # Check if any row has incomplete data
            incomplete_rows = False
            for index, row in edited_Tubing_df.iterrows():
                if row.isnull().any():
                    incomplete_rows = True
                    st.error(f"Please complete all fields in row {index+1} before saving.")
                    break
            
            if not incomplete_rows:
                st.session_state.Tubing = edited_Tubing_df
                st.session_state.tubing_edit_complete = True
                st.success("Tubing data saved successfully!")

    st.markdown("---")

    # The additional details section for Tubing
    if st.session_state.tubing_edit_complete:
        st.subheader('Additional Tubing Details')
        with st.form("additional_details_form_tubing"):
            if 'additional_data2' not in st.session_state:
                st.session_state.additional_data2 = {}
            
            for index, row in st.session_state.Tubing.iterrows():
                section_name = row['Name']
                st.subheader(f"Properties for: {section_name} Tubing")
                
                with st.container():
                    st.write(f'Tubing Properties')
                    
                    st.number_input(
                        'Density (ibm/ft3)',
                        min_value=0.0,
                        value=st.session_state.additional_data2.get(f'density_tubing_{index}', 0.0),
                        key=f'density_input_2{index}'
                    )
                    
                    st.number_input(
                        'Thermal Conductivity (BTU(h.degF.ft))',
                        min_value=0.0,
                        value=st.session_state.additional_data2.get(f'thermal_cond_tubing_{index}', 27.75),
                        key=f'thermal_cond_input_tubing_{index}'
                    )
                    
                    st.write('Annulus Material')
                    
                    st.number_input(
                        'fluid Denisty (Ibm/Gal)',
                        min_value=0.0,
                        value=st.session_state.additional_data2.get(f'fluid_denisty_{index}', 10.01449),
                        key=f'fluid_density_input_{index}'
                    )
                    
                    st.number_input(
                        'Fluid thermal conductivit (BTU(h.degF.ft)',
                        min_value=0.0,
                        value=st.session_state.additional_data2.get(f'fluid_thermal_cond_{index}', 0.58),
                        key=f'fluid_thermal_cond_input_{index}'
                    )
            
            submitted_details = st.form_submit_button("💾 Save Additional Details")
            
            if submitted_details:
                for index in range(len(st.session_state.Tubing)):
                    st.session_state.additional_data2[f'density_tubing_{index}'] = st.session_state[f'density_input_2{index}']
                    st.session_state.additional_data2[f'thermal_cond_tubing_{index}'] = st.session_state[f'thermal_cond_input_tubing_{index}']
                    st.session_state.additional_data2[f'fluid_denisty_{index}'] = st.session_state[f'fluid_density_input_{index}']
                    st.session_state.additional_data2[f'fluid_thermal_cond_{index}'] = st.session_state[f'fluid_thermal_cond_input_{index}']
                st.success("Additional details saved!")


# ------------------------------------------------
    # Completions
# ------------------------------------------------
# Completions Manager
#IPR function
def calculate_and_plot_ipr(completion_data, fluid_properties):
    """
    Calculates the IPR based on completion and fluid data and returns a matplotlib figure.
    Implements Jones's equation when selected: P_ws - P_wf = A * Q_L + B * Q_L^2
    """
    # Get data from completion and fluid
    ipr_model = completion_data['basic_info']['ipr_model']
    reservoir_pressure = completion_data['reservoir'].get('reservoir_pressure', 3000)
    productivity_index = completion_data['reservoir'].get('productivity_index', 1.0)
    use_vogel = completion_data['reservoir'].get('use_vogel_below_bubble_point', False)
    reservoir_temperature = completion_data['reservoir'].get('reservoir_temperature', 180)
    
    # Debug: Show what fluid properties we received
    st.write("### Debug Info")
    st.write(f"IPR Model: {ipr_model}")
    st.write(f"Reservoir Pressure: {reservoir_pressure} psi")
    st.write(f"Reservoir Temperature: {reservoir_temperature}°F")
    
    # Initialize variables
    pb = None
    aof = None
    
    # Generate a range of flowing bottomhole pressures (Pwf)
    pwf_values = np.linspace(100, reservoir_pressure, 50)
    q_values = []
    
    # Calculate flow rates for each Pwf based on IPR model
    if ipr_model == 'Vogel':
        # Get Vogel parameters from completion data
        q_max = completion_data['reservoir'].get('max_flow_rate', 1000.0)  # AOFP (Q_max)
        c = completion_data['reservoir'].get('vogel_coefficient', 0.2)     # Vogel coefficient
        
        st.write("### Vogel Model Parameters")
        st.write(f"AOFP (Q_max): {q_max} STB/D")
        st.write(f"Vogel Coefficient (C): {c}")
        
        # Calculate flow rates using Vogel's equation
        for pwf in pwf_values:
            pressure_ratio = pwf / reservoir_pressure
            # Vogel's equation: Q = Q_max * [1 - (1-C)*(Pwf/P_ws) - C*(Pwf/P_ws)^2]
            q = q_max * (1 - (1 - c) * pressure_ratio - c * pressure_ratio**2)
            q_values.append(max(0, q))
        
        # For Vogel model, AOF is the same as Q_max
        aof = q_max
    
    elif ipr_model == 'Fetkovitch':
        # Get Fetkovich parameters from completion data
        q_max = completion_data['reservoir'].get('max_flow_rate', 1000.0)  # AOFP (Q_max)
        n = completion_data['reservoir'].get('fetkovich_exponent', 1.0)   # Fetkovich exponent
        
        st.write("### Fetkovich Model Parameters")
        st.write(f"AOFP (Q_max): {q_max} STB/D")
        st.write(f"Fetkovich Exponent (n): {n}")
        
        # Calculate flow rates using Fetkovich's equation
        for pwf in pwf_values:
            pressure_ratio = pwf / reservoir_pressure
            # Fetkovich's equation: Q = Q_max * [1 - (Pwf/P_ws)^2]^n
            # Ensure we don't get negative values inside the power function
            base_value = max(0, 1 - pressure_ratio**2)
            q = q_max * (base_value)**n
            q_values.append(max(0, q))
        
        # For Fetkovich model, AOF is the same as Q_max
        aof = q_max
    
    elif ipr_model == 'Jones':
        # Get Jones parameters from completion data
        a = completion_data['reservoir'].get('jones_coefficient_a', 0.5)  # Laminar coefficient
        b = completion_data['reservoir'].get('jones_coefficient_b', 0.001)  # Turbulent coefficient
        
        st.write("### Jones Model Parameters")
        st.write(f"Coefficient A: {a} psi/STB/D")
        st.write(f"Coefficient B: {b} psi/(STB/D)^2")
        
        # Calculate flow rates using Jones's equation
        for pwf in pwf_values:
            pressure_drop = reservoir_pressure - pwf
            # Jones's equation: P_ws - P_wf = A * Q_L + B * Q_L^2
            # Rearranged to solve for Q_L: Q_L = [-A + sqrt(A^2 + 4*B*(P_ws-P_wf))] / (2*B)
            if b > 0:  # Avoid division by zero
                discriminant = a**2 + 4 * b * pressure_drop
                if discriminant >= 0:
                    q = (-a + np.sqrt(discriminant)) / (2 * b)
                else:
                    q = 0
            else:
                # If B is zero, equation reduces to linear: Q_L = (P_ws - P_wf) / A
                if a > 0:
                    q = pressure_drop / a
                else:
                    q = 0
            q_values.append(max(0, q))
        
        # Calculate AOF for Jones model (when P_wf = 0)
        if b > 0:
            discriminant = a**2 + 4 * b * reservoir_pressure
            if discriminant >= 0:
                aof = (-a + np.sqrt(discriminant)) / (2 * b)
            else:
                aof = 0
        else:
            if a > 0:
                aof = reservoir_pressure / a
            else:
                aof = 0
    
    elif ipr_model == 'Well PI' and use_vogel:
        # Composite PI-Vogel model (existing functionality)
        # Get fluid properties for bubble point calculation
        if fluid_properties:
            GOR = fluid_properties.get('GOR', 500)
            gas_specific_gravity = fluid_properties.get('gas_specific_gravity', 0.85)
            API = fluid_properties.get('API', 35)
            
            # Calculate Bubble Point Pressure using Standing Correlation
            try:
                gamma_o = 141.5 / (API + 131.5)
                pb_calculated = 18.2 * ((GOR * gas_specific_gravity) / gamma_o)**0.83 * \
                               10**(0.00091 * reservoir_temperature - 0.0125 * API)
                pb = max(100, min(pb_calculated, reservoir_pressure * 0.95))
                
                st.write(f"Calculated Bubble Point (Pb): {pb:.1f} psi")
            except Exception as e:
                pb = reservoir_pressure * 0.8
                st.error(f"Error calculating Pb: {e}. Using default: {pb} psi")
        else:
            pb = reservoir_pressure * 0.8
            st.write("No fluid properties available, using default Pb")
        
        # Calculate flow rates using composite PI-Vogel model
        for pwf in pwf_values:
            if pwf < pb:
                # Below bubble point - use Vogel equation
                qb = productivity_index * (reservoir_pressure - pb)
                q = qb + (productivity_index * pb / 1.8) * (1 - 0.2 * (pwf/pb) - 0.8 * (pwf/pb)**2)
            else:
                # Above bubble point - use straight-line PI
                q = productivity_index * (reservoir_pressure - pwf)
            
            q_values.append(max(0, q))
        
        # Calculate AOF for composite model
        aof = productivity_index * reservoir_pressure
    
    else:
        # Standard Straight-Line PI model (existing functionality)
        for pwf in pwf_values:
            q = productivity_index * (reservoir_pressure - pwf)
            q_values.append(max(0, q))
        
        # Calculate AOF for PI model
        aof = productivity_index * reservoir_pressure
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(q_values, pwf_values, 'b-', linewidth=2, label='IPR Curve')
    ax.set_xlabel('Flow Rate (STB/D)')
    ax.set_ylabel('Flowing Bottomhole Pressure, Pwf (psi)')
    ax.set_title(f"IPR Curve for {completion_data['basic_info']['name']}\n(Model: {ipr_model})")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add model-specific annotations
    if ipr_model == 'Vogel':
        # For Vogel model, mark reservoir pressure and AOFP
        ax.axhline(y=reservoir_pressure, color='k', linestyle=':', label=f'Reservoir Pressure ({reservoir_pressure} psi)')
        ax.axvline(x=q_max, color='g', linestyle='--', label=f'AOFP (Q_max = {q_max:.0f} STB/D)')
        
        # Add equation to the plot
        equation_text = r'$Q = Q_{max} \left[ 1 - (1-C)\frac{P_{wf}}{P_{ws}} - C\left(\frac{P_{wf}}{P_{ws}}\right)^2 \right]$'
        ax.text(0.5, 0.95, equation_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    elif ipr_model == 'Fetkovitch':
        # For Fetkovich model, mark reservoir pressure and AOFP
        ax.axhline(y=reservoir_pressure, color='k', linestyle=':', label=f'Reservoir Pressure ({reservoir_pressure} psi)')
        ax.axvline(x=q_max, color='g', linestyle='--', label=f'AOFP (Q_max = {q_max:.0f} STB/D)')
        
        # Add equation to the plot
        equation_text = r'$Q = Q_{max} \left[ 1 - \left(\frac{P_{wf}}{P_{ws}}\right)^2 \right]^n$'
        ax.text(0.5, 0.95, equation_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    elif ipr_model == 'Jones':
        # For Jones model, mark reservoir pressure and AOF
        ax.axhline(y=reservoir_pressure, color='k', linestyle=':', label=f'Reservoir Pressure ({reservoir_pressure} psi)')
        ax.axvline(x=aof, color='g', linestyle='--', label=f'AOF = {aof:.0f} STB/D')
        
        # Add equation to the plot
        equation_text = r'$P_{ws} - P_{wf} = A \cdot Q_L + B \cdot Q_L^2$'
        ax.text(0.5, 0.95, equation_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    elif ipr_model == 'Well PI' and use_vogel:
        # For composite PI-Vogel model, mark bubble point and reservoir pressure
        qb_value = productivity_index * (reservoir_pressure - pb)
        ax.axhline(y=pb, color='r', linestyle='--', label=f'Bubble Point (Pb = {pb:.0f} psi)')
        ax.axvline(x=qb_value, color='g', linestyle='--', label=f'Flow at Pb (Qb = {qb_value:.0f} STB/D)')
        ax.axhline(y=reservoir_pressure, color='k', linestyle=':', label=f'Reservoir Pressure ({reservoir_pressure} psi)')
    
    else:
        # For other models, just mark reservoir pressure
        ax.axhline(y=reservoir_pressure, color='k', linestyle=':', label=f'Reservoir Pressure ({reservoir_pressure} psi)')
    
    ax.legend()
    plt.tight_layout()
    return fig, pb, aof

if st.session_state.selected_tool == "Completions":
    st.subheader('Completions Manager 🔧')
    # Initialize completions data if not exists
    if 'completions' not in st.session_state:
        st.session_state.completions = {}

    if 'selected_completion' not in st.session_state:
        st.session_state.selected_completion = None

    if 'new_completion_mode' not in st.session_state:
        st.session_state.new_completion_mode = False

    # Main completions management
    if not st.session_state.selected_completion:
        if not st.session_state.new_completion_mode:
            # Show the add button
            if st.button("➕ Add New Completion", type='primary'):
                st.session_state.new_completion_mode = True
                st.rerun()
        else:
            # Show the input form for new completion
            st.subheader("Create New Completion")
            
            with st.form("new_completion_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    name = st.text_input('Completion Name', placeholder='Enter completion name')
                    geometry_profile = st.selectbox("Geometry Profile", ['Vertical', 'Horizontal', 'Deviated'])
                    fluid_entry = st.selectbox("Fluid Entry", ['Single point', 'Multiple points', 'Distributed'])
                
                with col2:
                    middle_md = st.number_input("Middle MD (ft)", min_value=0.0, step=1.0)
                    completion_type = st.selectbox('Type', ['Perforation', 'Open Hole', 'Slotted Liner', 'Screen'])
                    active = st.checkbox('Active', value=True)
                
                ipr_model = st.selectbox('IPR Model', [
                    'Well PI', 'Vogel', 'Fetkovitch', 'Jones'
                ])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.form_submit_button("💾 Create Completion", type='primary'):
                        if name.strip():
                            if name not in st.session_state.completions:
                                st.session_state.completions[name] = {
                                    'basic_info': {
                                        'name': name,
                                        'geometry_profile': geometry_profile,
                                        'fluid_entry': fluid_entry,
                                        'middle_md': middle_md,
                                        'type': completion_type,
                                        'active': active,
                                        'ipr_model': ipr_model,
                                        'created_date': datetime.now().strftime("%Y-%m-%d %H:%M")
                                    },
                                    'reservoir': {
                                        'productivity_index': 1.0,
                                        'use_vogel_below_bubble_point': False,
                                        'vogel_water_cut_correction': False
                                    },
                                    'sand': {},
                                    'fluid_model': {
                                        'selected_fluid': None,
                                        'flow_rate': 0.0,
                                        'pressure': 0.0,
                                        'temperature': 0.0
                                    },
                                    'notes': ''
                                }
                                st.success(f"✅ Completion '{name}' created successfully!")
                                st.session_state.new_completion_mode = False
                                st.rerun()
                            else:
                                st.error("❌ Completion name already exists!")
                        else:
                            st.error("❌ Please enter a valid completion name!")
                
                with col2:
                    if st.form_submit_button("❌ Cancel"):
                        st.session_state.new_completion_mode = False
                        st.rerun()

    # Display completions table - FIXED THIS LINE
    if 'completions' in st.session_state and st.session_state.completions:
        st.write('### Your Completions')
        
        for completion_name, completion_data in st.session_state.completions.items():
            basic_info = completion_data['basic_info']
            
            # Create a card-like display for each completion
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                
                with col1:
                    status_icon = "🟢" if basic_info['active'] else "🔴"
                    if st.button(f"{status_icon} {completion_name}", key=f"completion_{completion_name}"):
                        st.session_state.selected_completion = completion_name
                        st.rerun()
                
                with col2:
                    st.write(f"**Type:** {basic_info['type']}")
                    st.write(f"**MD:** {basic_info['middle_md']:.1f} ft")
                
                with col3:
                    st.write(f"**Profile:** {basic_info['geometry_profile']}")
                    st.write(f"**IPR:** {basic_info['ipr_model']}")
                
                with col4:
                    fluid_status = "✅" if completion_data['fluid_model']['selected_fluid'] else "❌"
                    st.write(f"**Fluid:** {fluid_status}")
                
                st.divider()

    # Completion Properties Editor
    if st.session_state.selected_completion:
        completion_name = st.session_state.selected_completion
        current_completion = st.session_state.completions[completion_name]
        
        st.subheader(f"Editing: {completion_name} 🔧")
        
        # Create tabs for different property categories
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Basic Info", "🏔️ Reservoir", "🏖️ Sand", "💧 Fluid Model"])
        
        # Tab 1: Basic Information
        with tab1:
            with st.form("basic_info_form"):
                st.write("### Basic Information")
                
                col1, col2 = st.columns(2)
                
                basic_info = current_completion['basic_info']
                
                with col1:
                    geometry_profile = st.selectbox(
                        "Geometry Profile",
                        ['Vertical', 'Horizontal', 'Deviated'],
                        index=['Vertical', 'Horizontal', 'Deviated'].index(basic_info['geometry_profile'])
                    )
                    
                    fluid_entry = st.selectbox(
                        "Fluid Entry",
                        ['Single point', 'Multiple points', 'Distributed'],
                        index=['Single point', 'Multiple points', 'Distributed'].index(basic_info['fluid_entry'])
                    )
                    
                    middle_md = st.number_input(
                        "Middle MD (ft)",
                        value=basic_info['middle_md'],
                        min_value=0.0,
                        step=1.0
                    )
                
                with col2:
                    completion_type = st.selectbox(
                        'Type',
                        ['Perforation'],
                        index=['Perforation'].index(basic_info['type'])
                    )
                    
                    active = st.checkbox('Active', value=basic_info['active'])
                    
                    ipr_model = st.selectbox(
                        'IPR Model',
                        ['Well PI', 'Vogel', 'Fetkovitch', 'Jones'],
                        index=['Well PI', 'Vogel', 'Fetkovitch', 'Jones'].index(basic_info['ipr_model'])
                    )
                
                if st.form_submit_button("💾 Update Basic Info", type="primary"):
                    current_completion['basic_info'].update({
                        'geometry_profile': geometry_profile,
                        'fluid_entry': fluid_entry,
                        'middle_md': middle_md,
                        'type': completion_type,
                        'active': active,
                        'ipr_model': ipr_model,
                        'last_modified': datetime.now().strftime("%Y-%m-%d %H:%M")
                    })
                    st.success("✅ Basic information updated!")
                    st.rerun()
        
        # Tab 2: Reservoir Data
        with tab2:
            with st.form("reservoir_form"):
                st.write("### Reservoir Properties")
              
                # Get current IPR model
                ipr_model = current_completion['basic_info']['ipr_model']
                col1, col2 = st.columns(2)
                
                with col1:
                    reservoir_pressure = st.number_input(
                        "Reservoir Pressure (psi)",
                        value=current_completion['reservoir'].get('reservoir_pressure', 3000.0),
                        min_value=0.0,
                        step=10.0
                    )
                
                with col2:
                    reservoir_temperature = st.number_input(
                        "Reservoir Temperature (°F)",
                        value=current_completion['reservoir'].get('reservoir_temperature', 180.0),
                        min_value=0.0,
                        step=1.0
                    )
                
                # IPR-specific inputs
                if ipr_model == 'Well PI':
                    st.write("#### Well PI Parameters")
                    
                    productivity_index = st.number_input(
                        "Productivity Index (stb/day/psi)",
                        value=current_completion['reservoir'].get('productivity_index', 1.0),
                        min_value=0.0,
                        step=0.1,
                        help="Productivity index for Well PI model"
                    )
                    
                    use_vogel_below_bubble_point = st.checkbox(
                        "Use Vogel below bubble point",
                        value=current_completion['reservoir'].get('use_vogel_below_bubble_point', False),
                        help="Apply Vogel correction when pressure falls below bubble point"
                    )
                    
                    # Store these values in the reservoir data
                    current_completion['reservoir']['productivity_index'] = productivity_index
                    current_completion['reservoir']['use_vogel_below_bubble_point'] = use_vogel_below_bubble_point
                
                elif ipr_model == 'Vogel':
                    st.write("#### Vogel Parameters")
                    
                    # Add AOFP (Absolute Open Flow Potential) input
                    aofp = st.number_input(
                        "AOFP - Absolute Open Flow Potential (STB/D)",
                        value=current_completion['reservoir'].get('max_flow_rate', 1000.0),
                        min_value=0.0,
                        step=10.0,
                        help="Maximum flow rate when bottomhole pressure is zero (Q_max in Vogel equation)"
                    )
                    
                    # Vogel coefficient input (0 to 1)
                    vogel_coefficient = st.number_input(
                        "Vogel Coefficient (C)",
                        value=current_completion['reservoir'].get('vogel_coefficient', 0.2),
                        min_value=0.0,
                        max_value=1.0,
                        step=0.01,
                        help="Vogel coefficient (typically 0.2 for solution gas drive reservoirs)"
                    )
                    
                    # Store the values - FIXED THE TYPO HERE
                    current_completion['reservoir']['max_flow_rate'] = aofp  # Changed from max_flow_rat to aofp
                    current_completion['reservoir']['vogel_coefficient'] = vogel_coefficient
                    
                    # Display Vogel equation
                    st.latex(r"Q = Q_{max} \left[ 1 - (1-C)\frac{P_{wf}}{P_{ws}} - C\left(\frac{P_{wf}}{P_{ws}}\right)^2 \right]")
                
                elif ipr_model == 'Fetkovitch':
                    st.write("#### Fetkovich Parameters")
                    
                    # Add AOFP (Absolute Open Flow Potential) input
                    aofp = st.number_input(
                        "AOFP - Absolute Open Flow Potential (STB/D)",
                        value=current_completion['reservoir'].get('max_flow_rate', 1000.0),
                        min_value=0.0,
                        step=10.0,
                        help="Maximum flow rate when bottomhole pressure is zero (Q_max in Fetkovich equation)"
                    )
                    
                    # Fetkovich exponent input
                    fetkovich_exponent = st.number_input(
                        "Fetkovich Exponent (n)",
                        value=current_completion['reservoir'].get('fetkovich_exponent', 1.0),
                        min_value=0.1,
                        max_value=3.0,
                        step=0.1,
                        help="Fetkovich exponent (dimensionless parameter)"
                    )
                    
                    # Store the values
                    current_completion['reservoir']['max_flow_rate'] = aofp
                    current_completion['reservoir']['fetkovich_exponent'] = fetkovich_exponent
                    
                    # Display Fetkovich equation
                    st.latex(r"Q = Q_{max} \left[ 1 - \left(\frac{P_{wf}}{P_{ws}}\right)^2 \right]^n")
                
                elif ipr_model == 'Jones':
                    st.write("#### Jones Parameters")
                    
                    # Add Jones coefficient A input
                    jones_a = st.number_input(
                        "Jones Coefficient A",
                        value=current_completion['reservoir'].get('jones_coefficient_a', 0.5),
                        min_value=0.0,
                        step=0.01,
                        format="%.4f",
                        help="Laminar flow coefficient (psi/STB/D)"
                    )
                    
                    # Add Jones coefficient B input
                    jones_b = st.number_input(
                        "Jones Coefficient B",
                        value=current_completion['reservoir'].get('jones_coefficient_b', 0.001),
                        min_value=0.0,
                        step=0.0001,
                        format="%.6f",
                        help="Turbulent flow coefficient (psi/(STB/D)^2)"
                    )
                    
                    # Store the values
                    current_completion['reservoir']['jones_coefficient_a'] = jones_a
                    current_completion['reservoir']['jones_coefficient_b'] = jones_b
                    
                    # Display Jones equation
                    st.latex(r"P_{ws} - P_{wf} = A \cdot Q_L + B \cdot Q_L^2")
                
                if st.form_submit_button("💾 Update Reservoir Data", type="primary"):
                    # Save the basic reservoir properties
                    current_completion['reservoir'].update({
                        'reservoir_pressure': reservoir_pressure,
                        'reservoir_temperature': reservoir_temperature
                    })
                    st.success("✅ Reservoir data updated!")
                    st.rerun()
            
            # --- IPR CALCULATION BUTTON OUTSIDE THE FORM ---
            st.divider()
            
            # This button is now independent of the form above
            if st.button("📈 Calculate & Plot IPR", type="primary"):
                # Get the selected fluid's properties
                selected_fluid_name = current_completion['fluid_model'].get('selected_fluid')
                fluid_props_for_calc = {}
                
                if selected_fluid_name and selected_fluid_name in st.session_state.fluids:
                    fluid_props_for_calc = st.session_state.fluids[selected_fluid_name].get('properties', {})
                else:
                    st.warning("No fluid selected for this completion. Using default values for Pb calculation if needed.")
                
                # Call the calculation function
                try:
                    ipr_fig, calculated_pb, aof = calculate_and_plot_ipr(current_completion, fluid_props_for_calc)
                    st.pyplot(ipr_fig)
                    
                    # Display key results
                    col1, col2 = st.columns(2)
                    
                    # Only show bubble point if it's not None
                    if calculated_pb is not None:
                        with col1:
                            st.metric("Calculated Bubble Point (Pb)", f"{calculated_pb:.1f} psi")
                    
                    # Only show AOF if it's not None
                    if aof is not None:
                        with col2:
                            st.metric("Absolute Open Flow (AOF)", f"{aof:.0f} STB/D")
                        
                    # Store these results if desired
                    if calculated_pb is not None:
                        current_completion['reservoir']['calculated_pb'] = calculated_pb
                    if aof is not None:
                        current_completion['reservoir']['aof'] = aof
                        
                except Exception as e:
                    st.error(f"An error occurred while calculating the IPR: {e}")
        
        # Tab 3: Sand Data
        with tab3:
            with st.form("sand_form"):
                st.write("### Sand Properties")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    sand_type = st.selectbox(
                        "Sand Type",
                        ["Consolidated", 'Unconsolidated', "Fractured", "Carbonate"],
                        index=["Consolidated", 'Unconsolidated', "Fractured", "Carbonate"].index(
                            current_completion['sand'].get('sand_type', 'Consolidated')
                        )
                    )
                    
                    grain_size = st.selectbox(
                        "Grain Size",
                        ["Fine", "Medium", "Coarse", "Very Coarse"],
                        index=["Fine", "Medium", "Coarse", "Very Coarse"].index(
                            current_completion['sand'].get('grain_size', 'Medium')
                        )
                    )
                    
                    sorting = st.selectbox(
                        "Sorting",
                        ["Well Sorted", "Moderately Sorted", "Poorly Sorted"],
                        index=["Well Sorted", "Moderately Sorted", "Poorly Sorted"].index(
                            current_completion['sand'].get('sorting', 'Moderately Sorted')
                        )
                    )
                
                with col2:
                    compaction_factor = st.slider(
                        "Compaction Factor",
                        min_value=0.0,
                        max_value=1.0,
                        value=current_completion['sand'].get('compaction_factor', 0.3),
                        step=0.01,
                        help="Sand compaction factor (0-1)"
                    )
                    
                    clay_content = st.slider(
                        "Clay Content (%)",
                        min_value=0.0,
                        max_value=50.0,
                        value=current_completion['sand'].get('clay_content', 10.0),
                        step=1.0
                    )
                    
                    sand_production_risk = st.selectbox(
                        "Sand Production Risk",
                        ["Low", "Medium", "High", "Critical"],
                        index=["Low", "Medium", "High", "Critical"].index(
                            current_completion['sand'].get('sand_production_risk', 'Medium')
                        )
                    )
                
                if st.form_submit_button("💾 Update Sand Data", type="primary"):
                    current_completion['sand'] = {
                        'sand_type': sand_type,
                        'grain_size': grain_size,
                        'sorting': sorting,
                        'compaction_factor': compaction_factor,
                        'clay_content': clay_content,
                        'sand_production_risk': sand_production_risk
                    }
                    st.success("✅ Sand data updated!")
                    st.rerun()
        
        # Tab 4: Fluid Model
        with tab4:
            with st.form("fluid_model_form"):
                st.write("### Fluid Model Configuration")
                
                # Check if fluids exist in fluid manager
                if 'fluids' not in st.session_state or not st.session_state.fluids:
                    st.warning("⚠️ No fluids available. Please create fluids in the Fluid Manager first.")
                    st.info("💡 Go to Fluid Manager → Add fluids → Return here to select them")
                else:
                    available_fluids = list(st.session_state.fluids.keys())
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Fluid selection
                        current_fluid = current_completion['fluid_model'].get('selected_fluid')
                        if current_fluid and current_fluid in available_fluids:
                            fluid_index = available_fluids.index(current_fluid)
                        else:
                            fluid_index = 0
                        
                        selected_fluid = st.selectbox(
                            "Select Fluid",
                            options=available_fluids,
                            index=fluid_index,
                            help="Choose from fluids created in Fluid Manager"
                        )
                    
                    # Display selected fluid properties
                    if selected_fluid:
                        st.write("#### Selected Fluid Properties")
                        fluid_props = st.session_state.fluids[selected_fluid].get('properties', {})
                        
                        if fluid_props:
                            prop_col1, prop_col2 = st.columns(2)
                            with prop_col1:
                                if 'water_cut' in fluid_props:
                                    st.write(f"**Water Cut:** {fluid_props['water_cut']:.3f}")
                                if 'GOR' in fluid_props:
                                    st.write(f"**GOR:** {fluid_props['GOR']:.3f} SCF/STB")
                            with prop_col2:
                                if 'water_specific_gravity' in fluid_props:
                                    st.write(f"**Water SG:** {fluid_props['water_specific_gravity']:.3f}")
                                if 'API' in fluid_props:
                                    st.write(f"**API Gravity:** {fluid_props['API']:.1f}")
                        else:
                            st.info("No properties defined for this fluid.")
                
                if st.form_submit_button("💾 Update Fluid Model", type="primary"):
                    if 'fluids' in st.session_state and st.session_state.fluids:
                        current_completion['fluid_model'] = {
                            'selected_fluid': selected_fluid,
                        }
                        st.success("✅ Fluid model updated!")
                        st.rerun()
                    else:
                        st.error("❌ No fluids available to assign!")
        
        # Navigation and action buttons
        st.divider()
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            if st.button("← Back to Completions List", type="secondary"):
                st.session_state.selected_completion = None
                st.rerun()
        
        with col2:
            # Add notes section
            with st.expander("📝 Notes"):
                notes = st.text_area(
                    "Completion Notes",
                    value=current_completion.get('notes', ''),
                    height=100,
                    key="completion_notes"
                )
                if st.button("💾 Save Notes"):
                    current_completion['notes'] = notes
                    st.success("Notes saved!")
                    st.rerun()
        
        with col3:
            if st.button("🗑️ Delete", type="secondary"):
                # Add confirmation
                if st.session_state.get('confirm_delete'):
                    del st.session_state.completions[completion_name]
                    st.session_state.selected_completion = None
                    if 'confirm_delete' in st.session_state:
                        del st.session_state.confirm_delete
                    st.success("Completion deleted!")
                    st.rerun()
                else:
                    st.session_state.confirm_delete = True
                    st.warning("Click delete again to confirm")
                    st.rerun()

    # Summary statistics - FIXED THIS LINE TOO
    if 'completions' in st.session_state and st.session_state.completions:
        st.divider()
        st.subheader("📊 Completions Summary")
        
        total_completions = len(st.session_state.completions)
        active_completions = sum(1 for comp in st.session_state.completions.values() if comp['basic_info']['active'])
        completions_with_fluids = sum(1 for comp in st.session_state.completions.values() if comp['fluid_model']['selected_fluid'])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Completions", total_completions)
        
        with col2:
            st.metric("Active Completions", active_completions)
        
        with col3:
            st.metric("With Assigned Fluids", completions_with_fluids)
        
        with col4:
            completion_rate = (completions_with_fluids / total_completions * 100) if total_completions > 0 else 0
            st.metric("Configuration Complete", f"{completion_rate:.0f}%")

            
#Well schematics
if st.session_state.selected_tool == "Well schematics":
    # Get bottom depth from survey section
    bottom_depth = st.session_state.get('bottom_depth', 0.0)
    survey_type = st.session_state.get('survey_type', 'Vertical')
    
    # Check if we have valid tubular data from the Tubulars section
    has_valid_casing_data = (
        'casing_liners' in st.session_state and 
        not st.session_state.casing_liners.empty and
        len(st.session_state.casing_liners) > 0 and
        st.session_state.casing_edit_complete  # Only use data if user has completed the edit
    )
    
    has_valid_tubing_data = (
        'Tubing' in st.session_state and 
        not st.session_state.Tubing.empty and
        len(st.session_state.Tubing) > 0 and
        st.session_state.tubing_edit_complete  # Only use data if user has completed the edit
    )
    
    if has_valid_casing_data or has_valid_tubing_data:
        # Verify that all required data is present and valid
        valid_casing_rows = []
        valid_tubing_rows = []
        
        # Process casing/liner data
        if has_valid_casing_data:
            for index, row in st.session_state.casing_liners.iterrows():
                # Check if all required fields have values (not NaN or empty)
                required_fields = ['Section type', 'Name', 'From MD', 'To MD', 'ID(in)', 'OD(in)']
                has_all_required = all(pd.notna(row[field]) for field in required_fields)
                
                if has_all_required:
                    valid_casing_rows.append(row)
        
        # Process tubing data
        if has_valid_tubing_data:
            for index, row in st.session_state.Tubing.iterrows():
                # Check if all required fields have values (not NaN or empty)
                required_fields = ['Name', 'To MD', 'ID(in)', 'OD(in)']
                has_all_required = all(pd.notna(row[field]) for field in required_fields)
                
                if has_all_required:
                    # For tubing, we need to set From MD to 0 (starts at surface)
                    row_with_from_md = row.copy()
                    row_with_from_md['From MD'] = 0.0
                    row_with_from_md['Section type'] = 'Tubing'
                    valid_tubing_rows.append(row_with_from_md)
        
        if valid_casing_rows or valid_tubing_rows:
            st.info("Using tubular data from Tubulars section")
            
            # Determine KOP based on survey type
            kop = None
            
            if survey_type == "Vertical":
                # For vertical wells, KOP equals bottom depth
                kop = bottom_depth
                st.info(f"Vertical well detected: Setting KOP to bottom depth ({bottom_depth} ft)")
            else:
                # For 2D/3D wells, find the first row where angle > 0
                if 'survey_df' in st.session_state and not st.session_state.survey_df.empty:
                    survey_df = st.session_state.survey_df
                    if 'Angle (°)' in survey_df.columns:
                        # Find the first row with angle > 0
                        first_angle_row = survey_df[survey_df['Angle (°)'] > 0].head(1)
                        if not first_angle_row.empty:
                            kop = first_angle_row['MD (ft)'].values[0]
                            st.info(f"{survey_type} well detected: Setting KOP to first angle change at {kop} ft")
                        else:
                            # If no angle > 0 found, use bottom depth
                            kop = bottom_depth
                            st.warning(f"{survey_type} well but no angle > 0 found. Setting KOP to bottom depth")
                    else:
                        kop = bottom_depth
                        st.warning("Angle column not found in survey data. Setting KOP to bottom depth")
                else:
                    kop = bottom_depth
                    st.warning("No survey data available. Setting KOP to bottom depth")
            
            # Create tubular objects from the valid data
            tubulars = []
            
            # Process casing/liner data
            for row in valid_casing_rows:
                section_type = row['Section type']
                name = row['Name']
                from_md = row['From MD']
                to_md = min(row['To MD'], bottom_depth) if bottom_depth > 0 else row['To MD']
                id_val = row['ID(in)']
                od_val = row['OD(in)']
                
                # Create appropriate tubular object based on section type
                if section_type in ['Casing', 'Liner']:
                    try:
                        tubular = Tubular(
                            name=name,
                            inD=id_val,
                            outD=od_val,
                            weight=0,  # Using 0 as default since weight isn't in the form
                            top=from_md,
                            low=to_md,
                            shoeSize=od_val  # Using OD as shoe size
                        )
                        tubulars.append(tubular)
                    except Exception as e:
                        st.error(f"Error creating {section_type} '{name}': {str(e)}")
                
                # Handle Open hole if needed
                elif section_type == 'Open hole':
                    # You might need to create a different object for open hole
                    st.warning(f"Open hole section '{name}' detected but not processed in schematic")
            
            # Process tubing data
            for row in valid_tubing_rows:
                section_type = row['Section type']
                name = row['Name']
                from_md = row['From MD']  # This is 0.0 as we set above
                to_md = min(row['To MD'], bottom_depth) if bottom_depth > 0 else row['To MD']
                id_val = row['ID(in)']
                od_val = row['OD(in)']
                
                # Create tubing object
                try:
                    tubing = Tubing(
                        name=name,
                        inD=id_val,
                        outD=od_val,
                        weight=0,  # Using 0 as default
                        top=from_md,
                        low=to_md
                    )
                    tubulars.append(tubing)
                    st.success(f"Added tubing '{name}' to schematic")
                except Exception as e:
                    st.error(f"Error creating tubing '{name}': {str(e)}")
            
            if tubulars:
                # Create and configure well with the determined KOP
                well = Well(name="Custom Well", kop=kop)
                
                # Add all tubular components to well
                for tubular in tubulars:
                    well.addTubular(tubular)
                
                st.success(f"Created well schematic with {len(tubulars)} tubular sections")
                st.info(f"KOP set to: {kop} ft")
                
                # Display summary of what was added
                casing_count = sum(1 for row in valid_casing_rows if row['Section type'] in ['Casing', 'Liner'])
                tubing_count = len(valid_tubing_rows)
                st.info(f"Includes: {casing_count} casing/liner sections, {tubing_count} tubing sections")
            else:
                st.warning("No valid tubular sections could be created from the data")
                st.stop()  # Stop execution if no valid tubulars
        else:
            st.warning("No complete tubular data found in the Tubulars section")
            st.stop()  # Stop execution if no valid rows
    else:
        st.warning("No tubular data available. Please complete the Tubulars section first.")
        st.info("Go to the Tubulars section to input your well configuration data.")
        st.stop()  # Stop execution if no tubular data
    
    # Display the bottom depth being used if available
    if bottom_depth > 0:
        st.info(f"Using bottom depth from survey: {bottom_depth} ft")
    
    # Create figure explicitly
    plt.figure()
    
    # Generate visualization
    try:
        well.visualize()
        
        # Get the current figure
        fig = plt.gcf()
        
        # Display the figure in Streamlit
        st.pyplot(fig)
        
        # Close the figure to free memory
        plt.close(fig)
    except Exception as e:
        st.error(f"Error generating visualization: {str(e)}")

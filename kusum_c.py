
# import streamlit as st
# import pandas as pd
# import numpy as np
# from io import BytesIO

# st.set_page_config(page_title="Substation Allocation Tool", layout="wide")

# # Title
# st.title("📊 Substation Allocation Tool")
# st.write(
#     """
#     Upload an Excel file as input (`combined_df`), preview & edit the data if needed, 
#     and process substation allocations iteratively. Navigate through iterations and download the processed Excel file.
#     """
# )

# # Upload Excel file
# uploaded_file = st.file_uploader("📂 Upload your Excel file", type=["xlsx"])

# # Initialize session state
# if "iteration_index" not in st.session_state:
#     st.session_state.iteration_index = 0
# if "iterations" not in st.session_state:
#     st.session_state.iterations = []
# if "combined_df" not in st.session_state:
#     st.session_state.combined_df = None

# # Function: Allocation logic
# def run_allocation(combined_df):
#     iterations_data = []
#     cumulative_alloted_capacity = 0.0
#     cumulative_new_ncfa = 0.0

#     df_result = pd.DataFrame(columns=[
#         'id', 'substation', 'project_capacity', 'tariff', 'bidder',
#         'alloted_capacity', 'new_ncfa', 'cumulative_alloted_capacity', 'cumulative_new_ncfa'
#     ])

#     iteration = 1
#     while True:
#         # STEP 1: Filter eligible rows
#         eligible = combined_df[
#             (combined_df['proj_cap'] <= combined_df['part_cap']) &
#             (combined_df['proj_cap'] <= combined_df['allo_cap'])
#         ]
#         if eligible.empty:
#             break

#         # STEP 2: Build df1 (lowest tariff, highest proj_cap)
#         min_tariff = eligible['tariff'].min()
#         low_tariff_df = eligible[eligible['tariff'] == min_tariff]
#         max_proj_cap = low_tariff_df['proj_cap'].max()
#         df1 = low_tariff_df[low_tariff_df['proj_cap'] == max_proj_cap].copy()

#         df1['remarks'] = np.where(
#             df1.shape[0] == 1,
#             "Direct Allotment",
#             "Lottery is required"
#         )
#         df1['lottery'] = np.nan

#         lottery_required = False

#         # Lottery if needed
#         if df1.shape[0] > 1:
#             lottery_required = True
#             df1['lottery'] = np.random.randint(1000, 9999, size=df1.shape[0])
#             winner_row = df1.loc[df1['lottery'].idxmin()]
#         else:
#             winner_row = df1.iloc[0]

#         # STEP 3: Build df2 (all bids for winner's substation)
#         winner_id = winner_row['id']
#         df2 = combined_df[combined_df['id'] == winner_id].copy()
#         df2 = df2.sort_values('tariff').reset_index(drop=True)

#         # Allocate capacities
#         df2['alloted_capacity'] = 0.0
#         df2['new_ncfa'] = np.minimum(df2['ncfa'], df2['qcfa'])
#         df2['remaining_participation_capacity'] = df2['part_cap'] - df2['proj_cap']
#         df2['remaining_allocable_capacity'] = df2['allo_cap'] - df2['alloted_capacity']
#         df2.loc[0, 'alloted_capacity'] = df2.loc[0, 'proj_cap']
#         df2.loc[0, 'remaining_allocable_capacity'] = df2.loc[0, 'allo_cap'] - df2.loc[0, 'alloted_capacity']

#         # Keep df2 as-is for display
#         df2 = df2[['id', 'substation', 'proj_cap', 'tariff', 'bidder',
#                    'part_cap', 'allo_cap', 'alloted_capacity', 'new_ncfa',
#                    'remaining_participation_capacity', 'remaining_allocable_capacity']]

#         # STEP 4: Update df_result
#         winner_df = df2.iloc[[0]].copy()
#         cumulative_alloted_capacity += winner_df['alloted_capacity'].values[0]
#         cumulative_new_ncfa += winner_df['new_ncfa'].values[0]
#         winner_df['cumulative_alloted_capacity'] = cumulative_alloted_capacity
#         winner_df['cumulative_new_ncfa'] = cumulative_new_ncfa

#         # Rename columns for df_result
#         winner_df.rename(columns={
#             'proj_cap': 'project_capacity',
#             'part_cap': 'participation_capacity',
#             'allo_cap': 'allocable_capacity'
#         }, inplace=True)

#         # Drop unnecessary columns from df_result
#         winner_df.drop(columns=[
#             'participation_capacity',
#             'allocable_capacity',
#             'remaining_participation_capacity',
#             'remaining_allocable_capacity'
#         ], inplace=True)

#         df_result = pd.concat([df_result, winner_df], ignore_index=True)

#         # Save iteration data for UI display
#         iterations_data.append({
#             "df1": df1,
#             "df2": df2,
#             "df_result": df_result.copy(),
#             "iteration": iteration,
#             "lottery_required": lottery_required,
#             "lottery_resolved": False
#         })

#         # STEP 5: Update combined_df
#         for _, row in df2.iterrows():
#             mask = (combined_df['bidder'] == row['bidder'])
#             combined_df.loc[mask, 'part_cap'] = row['remaining_participation_capacity']
#             combined_df.loc[mask, 'allo_cap'] = row['remaining_allocable_capacity']

#         combined_df = combined_df[~combined_df['id'].isin(df_result['id'])].reset_index(drop=True)
#         iteration += 1

#     return iterations_data

# # If file uploaded, show editable DataFrame
# if uploaded_file is not None:
#     st.session_state.combined_df = pd.read_excel(uploaded_file)
#     st.success("✅ File uploaded successfully!")

#     st.subheader("📝 Edit Uploaded Data (if needed)")
#     edited_df = st.data_editor(
#         st.session_state.combined_df,
#         use_container_width=True,
#         key="editable_combined_df"
#     )

#     # Run allocation
#     if st.button("🚀 Run Allocation Process"):
#         st.session_state.iterations = run_allocation(edited_df)
#         st.session_state.iteration_index = 0

# # Display iterations if available
# if st.session_state.iterations:
#     iter_idx = st.session_state.iteration_index
#     iteration_data = st.session_state.iterations[iter_idx]

#     st.header(f"📄 Allocation {iteration_data['iteration']}")

#     if iteration_data['lottery_required'] and not iteration_data['lottery_resolved']:
#         st.subheader("Substation (Before Lottery)")
#         st.dataframe(iteration_data["df1"].drop(columns=["lottery"]))
#         if st.button("🎲 Proceed with Lottery"):
#             st.session_state.iterations[iter_idx]["lottery_resolved"] = True
#             st.rerun()
#     else:
#         st.subheader("Substation")
#         st.dataframe(iteration_data["df1"])

#         st.subheader("Bids Against Substation")
#         st.dataframe(iteration_data["df2"])

#         st.subheader("Cumulative Allocation Results")
#         st.dataframe(iteration_data["df_result"])

#         # Navigation and Download
#         col1, col2, col3 = st.columns([1, 2, 1])
#         with col1:
#             if st.button("⬅️ Previous") and iter_idx > 0:
#                 st.session_state.iteration_index -= 1
#         with col3:
#             if st.button("Next ➡️") and iter_idx < len(st.session_state.iterations) - 1:
#                 st.session_state.iteration_index += 1
#         with col2:
#             output = BytesIO()
#             with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
#                 for i in range(iter_idx + 1):  # Only include iterations up to current
#                     iter_data = st.session_state.iterations[i]
#                     sheet_name = f"Allocation_{iter_data['iteration']}"
#                     iter_data["df1"].to_excel(writer, sheet_name=sheet_name, startrow=1, index=False)
#                     iter_data["df2"].to_excel(writer, sheet_name=sheet_name, startrow=len(iter_data["df1"]) + 4, index=False)
#                     iter_data["df_result"].to_excel(writer, sheet_name=sheet_name, startrow=len(iter_data["df1"]) + len(iter_data["df2"]) + 8, index=False)
#             output.seek(0)

#             st.download_button(
#                 label=f"📥 Download Excel (Up to Allocation {iter_idx+1})",
#                 data=output,
#                 file_name=f"allocation_iteration_{iter_idx+1}.xlsx",
#                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
#                 key=f"download_{iter_idx}"
#             )
############zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="Substation Allocation Tool", layout="wide")

# Title
st.title("📊 Substation Allocation Tool")
st.write(
    """
    Upload an Excel file as input (combined_df), preview & edit the data if needed, 
    and process substation allocations iteratively. Navigate through iterations and download the processed Excel file.
    """
)

# Upload Excel file
uploaded_file = st.file_uploader("📂 Upload your Excel file", type=["xlsx"])

# Initialize session state
if "iteration_index" not in st.session_state:
    st.session_state.iteration_index = 0
if "iterations" not in st.session_state:
    st.session_state.iterations = []
if "combined_df" not in st.session_state:
    st.session_state.combined_df = None

# Function: Allocation logic
def run_allocation(combined_df):
    iterations_data = []
    cumulative_alloted_capacity = 0.0
    cumulative_new_ncfa = 0.0

    df_result = pd.DataFrame(columns=[
        'id', 'circle', 'substation', 'project_capacity', 'tariff', 'bidder',
        'alloted_capacity', 'new_ncfa', 'cumulative_alloted_capacity', 'cumulative_new_ncfa'
    ])

    iteration = 1
    while True:
        # Stop if cumulative limits reached
        if cumulative_alloted_capacity > 200 or cumulative_new_ncfa > 100:
            break

        # STEP 1: Filter eligible rows
        eligible = combined_df[
            (combined_df['proj_cap'] <= combined_df['part_cap']) &
            (combined_df['proj_cap'] <= combined_df['allo_cap'])
        ]
        if eligible.empty:
            break

        # STEP 2: Build df1 (lowest tariff, highest proj_cap)
        min_tariff = eligible['tariff'].min()
        low_tariff_df = eligible[eligible['tariff'] == min_tariff]
        max_proj_cap = low_tariff_df['proj_cap'].max()
        df1 = low_tariff_df[low_tariff_df['proj_cap'] == max_proj_cap].copy()

        df1['remarks'] = np.where(
            df1.shape[0] == 1,
            "Direct Allotment",
            "Lottery is required"
        )
        df1['lottery'] = np.nan

        lottery_required = False

        # Lottery if needed
        if df1.shape[0] > 1:
            lottery_required = True
            df1['lottery'] = np.random.randint(1000, 9999, size=df1.shape[0])
            winner_row = df1.loc[df1['lottery'].idxmin()]
        else:
            winner_row = df1.iloc[0]

        # STEP 3: Build df2 (all bids for winner's substation)
        winner_id = winner_row['id']
        df2 = combined_df[combined_df['id'] == winner_id].copy()
        df2 = df2.sort_values('tariff').reset_index(drop=True)

        # Allocate capacities
        df2['alloted_capacity'] = 0.0
        df2['new_ncfa'] = np.minimum(df2['ncfa'], df2['qcfa'])
        df2['remaining_participation_capacity'] = df2['part_cap'] - df2['proj_cap']
        df2['remaining_allocable_capacity'] = df2['allo_cap'] - df2['alloted_capacity']
        df2.loc[0, 'alloted_capacity'] = df2.loc[0, 'proj_cap']
        df2.loc[0, 'remaining_allocable_capacity'] = df2.loc[0, 'allo_cap'] - df2.loc[0, 'alloted_capacity']

        # Keep df2 as-is for display
        df2 = df2[['id', 'circle', 'substation', 'proj_cap', 'tariff', 'bidder',
                   'part_cap', 'allo_cap', 'alloted_capacity', 'new_ncfa',
                   'remaining_participation_capacity', 'remaining_allocable_capacity']]

        # STEP 4: Update df_result
        winner_df = df2.iloc[[0]].copy()
        cumulative_alloted_capacity += winner_df['alloted_capacity'].values[0]
        cumulative_new_ncfa += winner_df['new_ncfa'].values[0]
        winner_df['cumulative_alloted_capacity'] = cumulative_alloted_capacity
        winner_df['cumulative_new_ncfa'] = cumulative_new_ncfa

        # Rename columns for df_result
        winner_df.rename(columns={
            'proj_cap': 'project_capacity',
            'part_cap': 'participation_capacity',
            'allo_cap': 'allocable_capacity'
        }, inplace=True)

        # Drop unnecessary columns from df_result
        winner_df.drop(columns=[
            'participation_capacity',
            'allocable_capacity',
            'remaining_participation_capacity',
            'remaining_allocable_capacity'
        ], inplace=True)

        df_result = pd.concat([df_result, winner_df], ignore_index=True)

        # Save iteration data for UI display
        iterations_data.append({
            "df1": df1,
            "df2": df2,
            "df_result": df_result.copy(),
            "iteration": iteration,
            "lottery_required": lottery_required,
            "lottery_resolved": False
        })

        # STEP 5: Update combined_df
        for _, row in df2.iterrows():
            mask = (combined_df['bidder'] == row['bidder'])
            combined_df.loc[mask, 'part_cap'] = row['remaining_participation_capacity']
            combined_df.loc[mask, 'allo_cap'] = row['remaining_allocable_capacity']

        combined_df = combined_df[~combined_df['id'].isin(df_result['id'])].reset_index(drop=True)
        iteration += 1

    return iterations_data

# If file uploaded, show editable DataFrame
if uploaded_file is not None:
    st.session_state.combined_df = pd.read_excel(uploaded_file)
    st.success("✅ File uploaded successfully!")

    st.subheader("📝 Edit Uploaded Data (if needed)")
    edited_df = st.data_editor(
        st.session_state.combined_df,
        use_container_width=True,
        key="editable_combined_df"
    )

    # Run allocation
    if st.button("🚀 Run Allocation Process"):
        st.session_state.iterations = run_allocation(edited_df)
        st.session_state.iteration_index = 0

# Display iterations if available
if st.session_state.iterations:
    iter_idx = st.session_state.iteration_index
    iteration_data = st.session_state.iterations[iter_idx]

    st.header(f"📄 Allocation {iteration_data['iteration']}")

    if iteration_data['lottery_required'] and not iteration_data['lottery_resolved']:
        st.subheader("Substation (Before Lottery)")
        st.dataframe(iteration_data["df1"].drop(columns=["lottery"]))
        if st.button("🎲 Proceed with Lottery"):
            st.session_state.iterations[iter_idx]["lottery_resolved"] = True
            st.rerun()
    else:
        st.subheader("Substation")
        st.dataframe(iteration_data["df1"])

        st.subheader("Bids Against Substation")
        st.dataframe(iteration_data["df2"])

        st.subheader("Cumulative Allocation Results")
        st.dataframe(iteration_data["df_result"])

        # Navigation and Download
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("⬅️ Previous") and iter_idx > 0:
                st.session_state.iteration_index -= 1
        with col3:
            if st.button("Next ➡️") and iter_idx < len(st.session_state.iterations) - 1:
                st.session_state.iteration_index += 1
        with col2:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                for i in range(iter_idx + 1):  # Only include iterations up to current
                    iter_data = st.session_state.iterations[i]
                    sheet_name = f"Allocation_{iter_data['iteration']}"
                    iter_data["df1"].to_excel(writer, sheet_name=sheet_name, startrow=1, index=False)
                    iter_data["df2"].to_excel(writer, sheet_name=sheet_name, startrow=len(iter_data["df1"]) + 4, index=False)
                    iter_data["df_result"].to_excel(writer, sheet_name=sheet_name, startrow=len(iter_data["df1"]) + len(iter_data["df2"]) + 8, index=False)
            output.seek(0)

            st.download_button(
                label=f"📥 Download Excel (Up to Allocation {iter_idx+1})",
                data=output,
                file_name=f"allocation_iteration_{iter_idx+1}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"download_{iter_idx}"
            )



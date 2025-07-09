
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# Title of the app
st.title("Substation Allocation Tool")
st.write(
    """
    📊 Upload an Excel file as input (`combined_df`) and download the processed allocation Excel file
    with all iterations in separate sheets.
    """
)

# Upload Excel file
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

# Run allocation when file is uploaded
if uploaded_file is not None:
    # Read the Excel file into DataFrame
    combined_df = pd.read_excel(uploaded_file)
    st.success("✅ File uploaded and read successfully.")

    # Display the uploaded DataFrame
    st.subheader("Preview of Uploaded Data")
    st.dataframe(combined_df.head())

    if st.button("Run Allocation Process"):
        st.info("⏳ Running allocation process. Please wait...")

        # Initialize trackers
        cumulative_alloted_cap = 0.0
        cumulative_new_ncfa = 0.0

        # Empty df_result to track winners
        df_result = pd.DataFrame(columns=[
            'id','substation','proj_cap','tariff','bidder',
            'alloted_cap','new_ncfa','remaining_part_cap','remaining_allo_cap',
            'cumulative_alloted_cap','cumulative_new_ncfa'
        ])

        iteration = 1
        output = BytesIO()

        # Main allocation loop
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            while True:
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
                    "Single Row, direct allotment of substation",
                    "Lottery is required"
                )
                df1['lottery'] = np.nan

                # Lottery if needed
                if df1.shape[0] > 1:
                    df1['lottery'] = np.random.randint(1000, 9999, size=df1.shape[0])
                    winner_row = df1.loc[df1['lottery'].idxmin()]
                else:
                    winner_row = df1.iloc[0]

                # STEP 3: Build df2 (all bids for winner's substation)
                winner_id = winner_row['id']
                df2 = combined_df[combined_df['id'] == winner_id].copy()
                df2 = df2.sort_values('tariff').reset_index(drop=True)

                # Allocate capacities
                df2['alloted_cap'] = 0.0
                df2['new_ncfa'] = np.minimum(df2['ncfa'], df2['qcfa'])
                df2['remaining_part_cap'] = df2['part_cap'] - df2['proj_cap']
                df2['remaining_allo_cap'] = df2['allo_cap'] - df2['alloted_cap']
                df2.loc[0, 'alloted_cap'] = df2.loc[0, 'proj_cap']
                df2.loc[0, 'remaining_allo_cap'] = df2.loc[0, 'allo_cap'] - df2.loc[0, 'alloted_cap']
                df2 = df2[['id','substation','proj_cap','tariff','bidder',
                           'part_cap','allo_cap','alloted_cap','new_ncfa',
                           'remaining_part_cap','remaining_allo_cap']]

                # STEP 4: Update df_result
                winner_df = df2.iloc[[0]].copy()
                cumulative_alloted_cap += winner_df['alloted_cap'].values[0]
                cumulative_new_ncfa += winner_df['new_ncfa'].values[0]
                winner_df['cumulative_alloted_cap'] = cumulative_alloted_cap
                winner_df['cumulative_new_ncfa'] = cumulative_new_ncfa
                df_result = pd.concat([df_result, winner_df], ignore_index=True)
                df_result=df_result[['id','substation','proj_cap','tariff','bidder','alloted_cap','new_ncfa','remaining_part_cap','remaining_allo_cap','cumulative_alloted_cap',	'cumulative_new_ncfa']]
                # STEP 5: Write to Excel
                sheet_name = f"Iteration_{iteration}"
                worksheet = writer.book.add_worksheet(sheet_name)
                writer.sheets[sheet_name] = worksheet

                worksheet.write(0, 0, "Substation")
                df1.to_excel(writer, sheet_name=sheet_name, startrow=1, index=False)

                df2_start_row = len(df1) + 4
                worksheet.write(df2_start_row - 1, 0, "Bids against substation")
                df2.to_excel(writer, sheet_name=sheet_name, startrow=df2_start_row, index=False)

                df_result_start_row = df2_start_row + len(df2) + 3
                worksheet.write(df_result_start_row - 1, 0, "Cumulative Allocation Results")
                df_result.to_excel(writer, sheet_name=sheet_name, startrow=df_result_start_row, index=False)

                # STEP 6: Update combined_df
                for _, row in df2.iterrows():
                    mask = (combined_df['bidder'] == row['bidder'])
                    combined_df.loc[mask, 'part_cap'] = row['remaining_part_cap']
                    combined_df.loc[mask, 'allo_cap'] = row['remaining_allo_cap']

                combined_df = combined_df[~combined_df['id'].isin(df_result['id'])].reset_index(drop=True)

                # Stop conditions
                if cumulative_alloted_cap >= 1500:
                    break
                if cumulative_new_ncfa >= 1200:
                    break

                iteration += 1

        # Prepare the Excel file for download
        output.seek(0)
        st.success("🎉 Allocation process completed!")

        # Provide download link
        st.download_button(
            label="📥 Download Allocation Excel File",
            data=output,
            file_name="allocation_iterations.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

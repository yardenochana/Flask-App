import pandas as pd
import re
from datetime import datetime
import numpy as np

def prepare_data(df):
    # Columns to drop
    columns_to_drop = ['Area', 'City', 'Pic_num', 'Cre_date', 'Repub_date', 'Color']
    df_dropped = df.drop(columns=columns_to_drop)
    
    # Standardize data by removing the 'manufactor' word from 'model' and removing years (numbers in parentheses)
    df_dropped['model'] = df_dropped.apply(lambda row: re.sub(r'\(\d{4}\)', '', row['model'].replace(row['manufactor'], '').strip()).strip(), axis=1)
    
    # Function to extract information from the 'Description' column
    def extract_info(description, unique_manufactors, unique_models):
        info = {}
        
        # Extract manufactor
        for manufactor in unique_manufactors:
            if re.search(manufactor, description):
                info['manufactor'] = manufactor
        
        # Extract Year
        year_match = re.search(r'שנה\s(198[0-9]|199[0-9]|200[0-9]|201[0-9]|202[0-4])\b', description)
        if year_match:
            info['Year'] = int(year_match.group(1))
        
        # Extract model
        for model in unique_models:
            if re.search(model, description, re.IGNORECASE):
                info['model'] = model
        
        # Extract Hand
        hand_match = re.search(r'\b(\d+)\s*יד\b', description)
        if hand_match:
            info['Hand'] = int(hand_match.group(1))
        
        # Extract Gear
        gears = ['אוטומטית', 'טיפטרוניק', 'ידנית', 'רובוטית', 'אוטומט', 'לא מוגדר']
        for gear in gears:
            if re.search(gear, description, re.IGNORECASE):
                info['Gear'] = gear
                break
        
        # Extract capacity_Engine
        capacity_engine_match = re.search(r'(?:(?:נפח|מנוע|נפח מנוע)\s*(\d+))', description)
        if capacity_engine_match:
            info['capacity_Engine'] = int(capacity_engine_match.group(1))
        
        # Extract Engine_type
        engine_types = ['בנזין', 'דיזל', 'גז', 'היברידי', 'היבריד', 'טורבו דיזל', 'חשמלי']
        for engine_type in engine_types:
            if re.search(engine_type, description, re.IGNORECASE):
                info['Engine_type'] = engine_type
                break
        
        return info

    def update_missing_values(row, columns_options, ownership_pattern):
        for column, options in columns_options.items():
            if pd.isna(row[column]):
                for option in options:
                    if option in row["Description"]:
                        if column == "Gear" and option in ["אוטומט", "אוטומטי"]:
                            row[column] = "אוטומטית"
                        elif column == "Engine_type":
                            if option == "היברידית":
                                row[column] = "היברידי"
                            elif option == "חשמלית":
                                row[column] = "חשמלי"
                            else:
                                row[column] = option
                        else:
                            row[column] = option
                        break
        
        # Update Prev_ownership and Curr_ownership if missing
        if pd.isna(row['Prev_ownership']) or pd.isna(row['Curr_ownership']):
            ownership_from_desc = find_ownership(row['Description'])
            if ownership_from_desc:
                if pd.isna(row['Prev_ownership']):
                    row['Prev_ownership'] = ownership_from_desc
                if pd.isna(row['Curr_ownership']):
                    row['Curr_ownership'] = ownership_from_desc
        
        return row

    def find_ownership(description):
        if pd.isna(description):
            return None
        match = re.search(ownership_pattern, description)
        if match:
            return match.group(0)
        return None

    def fill_values_from_description(df, unique_manufactors, unique_models, columns_options, ownership_pattern):
        for index, row in df.iterrows():
            if pd.notnull(row['Description']):
                extracted_info = extract_info(row['Description'], unique_manufactors, unique_models)
                for key, value in extracted_info.items():
                    if pd.isnull(row[key]):
                        df.at[index, key] = value
            update_missing_values(row, columns_options, ownership_pattern)
        return df

    # Define unique manufactors and models for extraction
    unique_manufactors = df['manufactor'].unique()
    unique_models = df['model'].unique()

    # Define columns options for missing values update
    columns_options = {
        'Gear': ['אוטומט', 'אוטומטי', 'ידני', 'רובוטי'],
        'Engine_type': ['בנזין', 'דיזל', 'היברידית', 'חשמלית']
    }

    # Define ownership pattern
    ownership_pattern = r'יד \d'
    
    # Define ownership types
    ownership_types = ['פרטית', 'השכרה', 'ליסינג', 'מונית', 'לימוד נהיגה', 'ייבוא אישי', 'ממשלתי']
    
    # Create ownership pattern
    ownership_pattern = '|'.join(ownership_types)
    
    # List of rows where missing values were completed
    completed_rows = []
    
    # Fill missing values in Prev_ownership and Curr_ownership
    for index, row in df_dropped.iterrows():
        if pd.isna(row['Prev_ownership']) or pd.isna(row['Curr_ownership']):
            ownership_from_desc = find_ownership(row['Description'])
            if ownership_from_desc:
                if pd.isna(row['Prev_ownership']):
                    df_dropped.at[index, 'Prev_ownership'] = ownership_from_desc
                if pd.isna(row['Curr_ownership']):
                    df_dropped.at[index, 'Curr_ownership'] = ownership_from_desc
                completed_rows.append(index)

    # Fill values from description
    df_dropped = fill_values_from_description(df_dropped, unique_manufactors, unique_models, columns_options, ownership_pattern)

    # Count missing values in each column
    missing_values = df_dropped.isnull().sum()
    
    # Calculate the percentage of missing values
    missing_percentage = (missing_values / len(df_dropped)) * 100
    
    # Create a DataFrame to display the results more nicely
    missing_values_df = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage})
    
    # Filter columns with more than 50% missing values
    columns_to_drop = missing_percentage[missing_percentage > 50].index
    
    # Drop the columns from df_dropped
    df_dropped = df_dropped.drop(columns=columns_to_drop)
    
    # Replace inconsistent values in 'model' column
    df_dropped['model'] = df_dropped['model'].replace({
        "קאונטרימן": "קאנטרימן",
        "גראנד, וויאגר": "גראנד, וויאג'ר",
        "גטה": "ג'טה",
        "גאז": "ג'אז",
        "C-Class קופה": "C-CLASS קופה",
        "E-CLASS": "E-Class",
        "E- CLASS": "E-Class"
    })
    
    # Replace inconsistent values in 'Gear' column
    df_dropped['Gear'] = df_dropped['Gear'].replace("אוטומט", "אוטומטית")
    
    # Replace inconsistent values in 'Engine_type' column
    df_dropped['Engine_type'] = df_dropped['Engine_type'].replace("היבריד", "היברידי")
    
    # Replace inconsistent values in 'manufactor' column
    df_dropped["manufactor"] = df_dropped["manufactor"].replace("Lexsus", "לקסוס")
    
    # Fill missing values in Year and Hand columns
    
    # Add a column Years_Since_Year indicating the number of years the car has been on the road
    current_year = datetime.now().year
    df_dropped['Years_Since_Year'] = current_year - df_dropped['Year']
    
    # Step 1: Find rows with no missing values in Hand and Years_Since_Year columns
    valid_rows = df_dropped.dropna(subset=['Hand', 'Years_Since_Year'])
    
    # Step 2: Calculate the ratio and mean ratio
    ratios = valid_rows['Years_Since_Year'] / valid_rows['Hand']
    mean_ratio = ratios.mean()
    
    # Step 3: Fill missing values in Hand column
    df_dropped['Hand'] = df_dropped.apply(
        lambda row: row['Years_Since_Year'] / mean_ratio if pd.isnull(row['Hand']) else row['Hand'],
        axis=1
    )
    
    # Step 4: Fill missing values in Years_Since_Year and Year columns
    df_dropped['Years_Since_Year'] = df_dropped.apply(
        lambda row: row['Hand'] * mean_ratio if pd.isnull(row['Years_Since_Year']) else row['Years_Since_Year'],
        axis=1
    )
    df_dropped['Year'] = df_dropped.apply(
        lambda row: current_year - row['Years_Since_Year'] if pd.isnull(row['Year']) else row['Year'],
        axis=1
    )
    
    # Fill missing values in Gear column
    
    # Create a dictionary with the most common values in the Gear column by year
    gear_mode_by_year = df_dropped.groupby('Year')['Gear'].agg(lambda x: x.mode()[0] if not x.mode().empty else None).to_dict()

    # Define a set of values considered as "undefined"
    undefined_values = {'לא מוגדר', None, np.nan}

    # Fill missing values and undefined values in Gear column according to the gear_mode_by_year dictionary
    df_dropped['Gear'] = df_dropped.apply(
        lambda row: gear_mode_by_year[row['Year']] if row['Gear'] in undefined_values else row['Gear'],
        axis=1
    )

    # Fill missing values in capacity_Engine column

    # Convert 'capacity_Engine' column to numeric, setting errors='coerce' to convert invalid parsing to NaN
    df_dropped['capacity_Engine'] = pd.to_numeric(df_dropped['capacity_Engine'], errors='coerce')

    # Replace empty string values with NaN
    df_dropped['capacity_Engine'] = df_dropped['capacity_Engine'].replace('', pd.NA)

    # Check for NaN values in 'capacity_Engine' column
    missing_values_before = df_dropped['capacity_Engine'].isna().sum()

    # Calculate the median value
    median_capacity = df_dropped['capacity_Engine'].median()

    # Replace NaN values with the median value
    df_dropped['capacity_Engine'] = df_dropped['capacity_Engine'].fillna(median_capacity)

    # Check for NaN values after filling
    missing_values_after = df_dropped['capacity_Engine'].isna().sum()

    # Identify outliers using IQR
    Q1 = df_dropped['capacity_Engine'].quantile(0.25)
    Q3 = df_dropped['capacity_Engine'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Handle outliers without changing other data
    df_dropped['capacity_Engine'] = df_dropped['capacity_Engine'].apply(lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x))

    # Calculate statistical data for the 'capacity_Engine' column after handling outliers
    capacity_engine_stats = df_dropped['capacity_Engine'].describe(percentiles=[.25, .5, .75])

    # Fill missing values in capacity_Engine column
    
    # Convert 'capacity_Engine' column to numeric
    df_dropped['capacity_Engine'] = pd.to_numeric(df_dropped['capacity_Engine'], errors='coerce')
    
    # Convert non-numeric values in the 'capacity_Engine' column to numeric, setting errors='coerce' to convert invalid parsing to NaN
    df['capacity_Engine'] = pd.to_numeric(df['capacity_Engine'], errors='coerce')

    # Replace NaN values with the median value
    median_capacity = df['capacity_Engine'].median()
    df['capacity_Engine'] = df['capacity_Engine'].fillna(median_capacity)
    
    # Fill missing values in Engine_type column
    
    # Create a dictionary with the most common values in the Engine_type column by manufactor
    engine_type_mode_by_manufactor = df_dropped.groupby('manufactor')['Engine_type'].agg(lambda x: x.mode()[0] if not x.mode().empty else None).to_dict()
    
    # Fill missing values in Engine_type column according to the engine_type_mode_by_manufactor dictionary
    df_dropped['Engine_type'] = df_dropped.apply(lambda row: engine_type_mode_by_manufactor[row['manufactor']] if pd.isnull(row['Engine_type']) else row['Engine_type'], axis=1)

    # Convert 'Km' column to string to use .str accessor
    df_dropped['Km'] = df_dropped['Km'].astype(str)

    # Remove commas and replace 'None' with NaN
    df_dropped['Km'] = df_dropped['Km'].str.replace(',', '').replace('None', np.nan)

    # Remove rows with non-convertible values
    df_dropped = df_dropped[pd.to_numeric(df_dropped['Km'], errors='coerce').notnull()]

    # Convert to float and then to int
    df_dropped['Km'] = df_dropped['Km'].astype(float).astype(int)

    # Fill missing values in 'Km' column with 0
    df_dropped['Km'] = df_dropped['Km'].fillna(0).astype(int)

    # Remove rows with Km value of 1000000
    df_dropped = df_dropped[df_dropped['Km'] != 1000000]

    # Multiply values less than 500 by 1000
    df_dropped.loc[df_dropped['Km'] < 500, 'Km'] *= 1000

    # Filter out 0 values and missing values in 'Km' column
    filtered_km = df_dropped[df_dropped['Km'] != 0]['Km']

    # Calculate the sum of values in 'Km' column after filtering
    sum_km = filtered_km.sum()

    # Calculate the sum of the difference in years
    sum_year_difference = df_dropped['Years_Since_Year'].sum()

    # Calculate new values in 'Km' column based on the requested formula
    df_dropped['Km'] = df_dropped.apply(lambda row: sum_km / sum_year_difference * row['Years_Since_Year'], axis=1)

    # Convert 'Km' column to string
    df_dropped['Km'] = df_dropped['Km'].astype(str)

    # Remove 'Years_Since_Year' column
    df_dropped.drop(columns=['Years_Since_Year'], inplace=True)
    
    # Ownership ranking dictionary
    ownership_ranking = {
        'מונית': 1,
        'לימוד נהיגה': 2,
        'השכרה': 3,
        'ליסינג': 4,
        'פרטית': 5,
        'ייבוא אישי': 6,
        'ממשלתי': 7
    }

    # Function to combine ownership values based on specified conditions
    def combine_ownership(prev, curr):
        if pd.isna(prev) and pd.isna(curr):
            return None
        if pd.isna(prev):
            return curr
        if pd.isna(curr):
            return prev
        if prev == curr:
            return prev
        return prev if ownership_ranking.get(prev, float('inf')) < ownership_ranking.get(curr, float('inf')) else curr

    # Combine the columns
    df_dropped['ownership'] = df_dropped.apply(lambda row: combine_ownership(row['Prev_ownership'], row['Curr_ownership']), axis=1)

    # Replace certain values with NaN
    df_dropped['ownership'].replace(['None', 'לא מוגדר', 'אחר'], pd.NA, inplace=True)

    # Fill missing values with the most common value
    most_common_ownership = df_dropped['ownership'].mode()[0]
    df_dropped['ownership'].fillna(most_common_ownership, inplace=True)

    # Remove 'Description' column
    df_dropped.drop(columns=['Prev_ownership', 'Curr_ownership', 'Description'], inplace=True)
    
    return df_dropped
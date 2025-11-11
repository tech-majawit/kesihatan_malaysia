import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- Configuration and Data Generation ---

@st.cache_data
def generate_mock_data(n_rows=15000):
    """Generates a synthetic dataset mimicking Malaysian communicable disease data."""
    np.random.seed(42)

    # Demographic/Geographic Data
    states = ['Johor', 'Kedah', 'Kelantan', 'Malacca', 'Negeri Sembilan', 'Pahang',
              'Penang', 'Perak', 'Perlis', 'Sabah', 'Sarawak', 'Selangor',
              'Terengganu', 'Kuala Lumpur', 'Putrajaya', 'Labuan']
    age_groups = ['0-4', '5-14', '15-29', '30-49', '50-64', '65+']
    genders = ['Male', 'Female']
    ethnicities = ['Malay', 'Chinese', 'Indian', 'Others']
    area_types = ['Urban', 'Rural']
    years = [2021, 2022, 2023]

    # Disease Data (Simplified categories/diseases)
    disease_map = {
        'Vector-borne': ['Dengue', 'Malaria', 'Chikungunya'],
        'Food/Water-borne': ['Typhoid Fever', 'Cholera', 'Hepatitis A'],
        'Respiratory': ['Tuberculosis', 'Influenza', 'COVID-19'],
        'Vaccine-Preventable': ['Measles', 'Mumps', 'Rubella'],
        'Zoonotic': ['Leptospirosis', 'Rabies']
    }
    all_diseases = [d for sublist in disease_map.values() for d in sublist]
    disease_categories = list(disease_map.keys())

    data = {
        'Year': np.random.choice(years, n_rows),
        'State': np.random.choice(states, n_rows),
        'Age_Group': np.random.choice(age_groups, n_rows),
        'Gender': np.random.choice(genders, n_rows),
        'Ethnicity': np.random.choice(ethnicities, n_rows),
        'Area_Type': np.random.choice(area_types, n_rows),
        'Disease': np.random.choice(all_diseases, n_rows)
    }

    df = pd.DataFrame(data)

    # Assign Category based on Disease
    category_list = []
    for disease in df['Disease']:
        for category, diseases in disease_map.items():
            if disease in diseases:
                category_list.append(category)
                break
        else:
            category_list.append('Other') # Should not happen with the current setup
    df['Disease_Category'] = category_list

    # Generate Metrics (Cases, Incidence, Mortality)
    df['Cases'] = np.random.randint(1, 100, n_rows)
    df['Population'] = np.random.randint(50000, 500000, n_rows) # Mock population base

    # Calculate Rates
    # Incidence Rate (per 100,000 population)
    df['Incidence_Rate'] = (df['Cases'] / df['Population']) * 100000
    # Mortality Rate (Cases * random factor, scaled)
    df['Mortality_Rate'] = (df['Cases'] * np.random.uniform(0.01, 0.5, n_rows)) / df['Population'] * 10000

    # Ensure all columns exist before returning
    df = df[['Year', 'State', 'Disease_Category', 'Disease', 'Age_Group', 'Gender',
             'Ethnicity', 'Area_Type', 'Cases', 'Incidence_Rate', 'Mortality_Rate']]
    
    return df

# Load the data
df_raw = generate_mock_data()
initial_data_count = len(df_raw)

# --- Streamlit Application ---

def run_dashboard():
    st.set_page_config(
        page_title="🇲🇾 Malaysia Official Health Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🇲🇾 Malaysia Official Health Surveillance Dashboard")
    st.markdown("---")

    # --- Sidebar Filters ---
    st.sidebar.header("Data Filters")

    # Filter 1: Disease Category (Single Select - acts as main selector)
    all_categories = df_raw['Disease_Category'].unique().tolist()
    selected_category = st.sidebar.selectbox(
        "Select Major Disease Category",
        options=['All Categories'] + all_categories,
        index=0
    )

    df_filtered = df_raw.copy()

    if selected_category != 'All Categories':
        df_filtered = df_filtered[df_filtered['Disease_Category'] == selected_category]
        available_diseases = sorted(df_filtered['Disease'].unique().tolist())
    else:
        available_diseases = sorted(df_raw['Disease'].unique().tolist())


    # Filter 2: Disease (Multi-Select, dependent on Category)
    selected_diseases = st.sidebar.multiselect(
        f"Select Specific Disease(s) (Total: {len(available_diseases)})",
        options=available_diseases,
        default=available_diseases
    )

    df_filtered = df_filtered[df_filtered['Disease'].isin(selected_diseases)]

    # Filter 3: View Type (Metric Selector)
    view_types = {
        "Total Cases (Count)": "Cases",
        "Incidence Rate (per 100k)": "Incidence_Rate",
        "Mortality Rate (per 10k)": "Mortality_Rate"
    }
    selected_view_name = st.sidebar.radio("Select View Metric", list(view_types.keys()), index=0)
    selected_metric = view_types[selected_view_name]

    # Demographic Filters (Expanders for cleaner look)
    with st.sidebar.expander("Geographic & Demographic Filters"):
        selected_states = st.multiselect(
            "Filter by State",
            options=sorted(df_raw['State'].unique().tolist()),
            default=sorted(df_raw['State'].unique().tolist())
        )
        selected_age = st.multiselect(
            "Filter by Age Group",
            options=df_raw['Age_Group'].unique().tolist(),
            default=df_raw['Age_Group'].unique().tolist()
        )
        selected_gender = st.multiselect(
            "Filter by Gender",
            options=df_raw['Gender'].unique().tolist(),
            default=df_raw['Gender'].unique().tolist()
        )
        selected_ethnicity = st.multiselect(
            "Filter by Ethnicity",
            options=df_raw['Ethnicity'].unique().tolist(),
            default=df_raw['Ethnicity'].unique().tolist()
        )
        selected_area = st.multiselect(
            "Filter by Area Type (Urban/Rural)",
            options=df_raw['Area_Type'].unique().tolist(),
            default=df_raw['Area_Type'].unique().tolist()
        )
        selected_years = st.multiselect(
            "Filter by Year",
            options=sorted(df_raw['Year'].unique().tolist()),
            default=sorted(df_raw['Year'].unique().tolist())
        )

    # Apply remaining filters
    df_filtered = df_filtered[
        (df_filtered['State'].isin(selected_states)) &
        (df_filtered['Age_Group'].isin(selected_age)) &
        (df_filtered['Gender'].isin(selected_gender)) &
        (df_filtered['Ethnicity'].isin(selected_ethnicity)) &
        (df_filtered['Area_Type'].isin(selected_area)) &
        (df_filtered['Year'].isin(selected_years))
    ]

    # --- Main Content and Visualization ---

    st.subheader(f"📊 {selected_view_name} Analysis")

    # Selector for Grouping/Breakdown
    grouping_options = {
        'State': 'State', 'Age Group': 'Age_Group', 'Gender': 'Gender',
        'Ethnicity': 'Ethnicity', 'Area Type': 'Area_Type', 'Year': 'Year'
    }
    selected_grouping_name = st.selectbox(
        "Breakdown Chart By:",
        options=list(grouping_options.keys()),
        index=0,
        key='grouping_selector'
    )
    grouping_column = grouping_options[selected_grouping_name]

    if df_filtered.empty:
        st.warning("No data found for the selected filters. Please adjust the sidebar selections.")
    else:
        # Group and aggregate data for visualization
        df_grouped = df_filtered.groupby(grouping_column)[selected_metric].sum().reset_index()
        
        # Sort by metric descending
        df_grouped = df_grouped.sort_values(by=selected_metric, ascending=False)
        
        # Format for display (for cleaner tooltips/labels)
        title_metric = selected_metric.replace('_', ' ').title()
        
        # Create Bar Chart
        fig = px.bar(
            df_grouped,
            x=grouping_column,
            y=selected_metric,
            text=selected_metric,
            labels={grouping_column: selected_grouping_name, selected_metric: title_metric},
            color=grouping_column, # Color by the grouping column
            template="plotly_white",
            title=f"{title_metric} Breakdown by {selected_grouping_name}"
        )

        # Update layout for better readability and label formatting
        fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        
        if selected_metric == 'Cases':
             fig.update_traces(texttemplate='%{y:,.0f}') # Format cases as integers
        elif selected_metric.endswith('Rate'):
             fig.update_traces(texttemplate='%{y:.2f}') # Format rates to 2 decimal places

        st.plotly_chart(fig, use_container_width=True)

        # --- Data Summary Section ---
        with st.expander("View Filtered Raw Data & Statistics"):
            col1, col2 = st.columns(2)
            col1.metric("Total Initial Records", initial_data_count)
            col2.metric("Filtered Records", len(df_filtered))

            st.dataframe(df_filtered.head(100), use_container_width=True) # Show first 100 rows

# Run the app
if __name__ == "__main__":
    run_dashboard()
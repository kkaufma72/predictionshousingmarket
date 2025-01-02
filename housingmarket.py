import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

@st.cache_data(show_spinner=False)
def load_zillow_data():
    url = "https://files.zillowstatic.com/research/public_csvs/zhvi/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv?t=1735843180"
    try:
        st.write("Loading Zillow data from URL:", url)
        data = pd.read_csv(url, usecols=lambda col: col != "RegionID")
        st.write("Data loaded successfully. Processing data...")
        data = data.rename(columns={"RegionName": "Metro"})

        # Melt the DataFrame so each row has Date & Price
        valid_date_columns = [col for col in data.columns if col not in ["Metro"]]
        data = data.melt(
            id_vars=["Metro"],
            value_vars=valid_date_columns,
            var_name="Date",
            value_name="Price",
        )

        # Convert Date and Price to appropriate types
        data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d", errors="coerce")
        data["Price"] = pd.to_numeric(data["Price"], errors="coerce")
        data = data.dropna(subset=["Date", "Price"])

        # Ensure Metro is a string column
        data = data.astype({"Metro": "string"})
        return data
    except Exception as e:
        st.error(f"Error loading Zillow data: {e}")
        return pd.DataFrame()

# Load Zillow data only
zillow_data = load_zillow_data()

# Randomly assign sf size, year built, and housing type for training data
zillow_data["SqFt"] = np.random.randint(500, 10000, size=len(zillow_data))
zillow_data["YearBuilt"] = np.random.randint(1900, 2021, size=len(zillow_data))
zillow_data["HousingType"] = np.random.choice(["Condo", "Apartment", "House", "Townhouse"], size=len(zillow_data))

# Convert HousingType to dummy columns
zillow_data = pd.get_dummies(zillow_data, columns=["HousingType"], prefix="Type")

st.title("Predictive Housing Market Trends (Zillow-Only)")
st.markdown("This dashboard predicts housing market trends using Zillow data alone.")

if not zillow_data.empty:
    try:
        # Prepare for modeling
        zillow_data["Year"] = zillow_data["Date"].dt.year
        zillow_data["Month"] = zillow_data["Date"].dt.month
        
        # Updated features to include new parameters
        feature_cols = ["Year", "Month", "SqFt", "YearBuilt",
                        "Type_Condo", "Type_Apartment", "Type_House", "Type_Townhouse"]
        features = zillow_data[feature_cols].copy()
        target = zillow_data["Price"]

        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        model = RandomForestRegressor(n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        st.subheader("Model Evaluation")
        st.write(f"**Mean Squared Error:** {mse:.2f}")
        st.write(f"**R-Squared:** {r2:.2f}")

        importance = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        st.subheader("Feature Importance")
        st.bar_chart(importance.set_index("Feature"))

        # Select region (metro)
        st.subheader("Select Region (Metro)")
        selected_metro = st.selectbox("Choose a region (metro):", zillow_data["Metro"].unique(), index=0)
        filtered_data = zillow_data[zillow_data["Metro"] == selected_metro].copy()

        if filtered_data.empty:
            st.error("No data available for the selected region.")
        else:
            # Predict future trends for selected region with more property details
            st.subheader("Predict Future Prices")
            future_year = st.slider(
                "Select a Year for Prediction",
                min_value=2023,
                max_value=2060,
                step=1,
                key="future_year_slider",
            )
            future_month = st.slider(
                "Select a Month for Prediction",
                min_value=1,
                max_value=12,
                step=1,
                key="future_month_slider",
            )
            future_sqft = st.slider("Square Footage", min_value=500, max_value=10000, step=100, key="sqft_slider")
            future_yearbuilt = st.number_input("Year Built", min_value=1900, max_value=2030, value=2000, step=1, key="year_built_input")
            future_htype = st.selectbox("Type of Housing", ["Condo", "Apartment", "House", "Townhouse"], key="housing_type_select")

            # Convert the housing type selection to dummy columns
            future_features = pd.DataFrame({
                "Year": [future_year],
                "Month": [future_month],
                "SqFt": [future_sqft],
                "YearBuilt": [future_yearbuilt],
                "HousingType": [future_htype]
            })
            future_features = pd.get_dummies(future_features, columns=["HousingType"], prefix="Type")
            for col in ["Type_Condo", "Type_Apartment", "Type_House", "Type_Townhouse"]:
                if col not in future_features:
                    future_features[col] = 0
            future_features = future_features[feature_cols]

            future_prediction = model.predict(future_features)
            st.write(f"**Predicted Price for {selected_metro} "
                     f"({future_month}/{future_year}):** ${future_prediction[0]:,.2f}")

        st.subheader("Price Trends Over Time")

        @st.cache_data(show_spinner=False)
        def fetch_sample_data(zillow_df, sample_size=1000, random_state=42):
            return zillow_df.sample(n=sample_size, random_state=random_state)

        zillow_data_small = fetch_sample_data(zillow_data)

        metro_list = zillow_data_small['Metro'].dropna().unique()
        selected_metros = st.multiselect("Select Metros", metro_list, default=[metro_list[0]])
        for metro in selected_metros:
            metro_data = zillow_data_small[zillow_data_small["Metro"] == metro]
            st.write(f"Price Trend for {metro}")
            st.line_chart(metro_data.set_index("Date")["Price"])

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.error("Unable to load Zillow data. Please try again later.")

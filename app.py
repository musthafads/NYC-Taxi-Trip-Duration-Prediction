
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium

from streamlit_folium import st_folium


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="NYC Taxi Trip Predictor",
    page_icon="🚕",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 25px;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "taxi_trip_duration_model.pkl"
    )

    feature_columns = joblib.load(
        "taxi_feature_columns.pkl"
    )

    try:

        model_info = joblib.load(
            "taxi_model_info.pkl"
        )

    except:

        model_info = {}

    return model, feature_columns, model_info


try:

    model, feature_columns, model_info = load_model()

except FileNotFoundError:

    st.error(
        "❌ Model files are missing."
    )

    st.write(
        "Make sure these files are in the same folder as app.py:"
    )

    st.code("""
taxi_trip_duration_model.pkl
taxi_feature_columns.pkl
taxi_model_info.pkl
    """)

    st.stop()


# =========================================================
# NYC LOCATION DATA
# =========================================================

locations = {

    "JFK Airport": {
        "id": 132,
        "lat": 40.6413,
        "lon": -73.7781
    },

    "LaGuardia Airport": {
        "id": 138,
        "lat": 40.7769,
        "lon": -73.8740
    },

    "Times Square": {
        "id": 230,
        "lat": 40.7580,
        "lon": -73.9855
    },

    "Midtown Manhattan": {
        "id": 161,
        "lat": 40.7549,
        "lon": -73.9840
    },

    "Central Park": {
        "id": 43,
        "lat": 40.7829,
        "lon": -73.9654
    },

    "Upper East Side": {
        "id": 236,
        "lat": 40.7736,
        "lon": -73.9566
    },

    "Upper West Side": {
        "id": 239,
        "lat": 40.7870,
        "lon": -73.9754
    },

    "Chelsea": {
        "id": 33,
        "lat": 40.7465,
        "lon": -74.0014
    },

    "SoHo": {
        "id": 206,
        "lat": 40.7233,
        "lon": -74.0000
    },

    "Tribeca": {
        "id": 261,
        "lat": 40.7163,
        "lon": -74.0086
    },

    "Financial District": {
        "id": 148,
        "lat": 40.7075,
        "lon": -74.0113
    },

    "Greenwich Village": {
        "id": 100,
        "lat": 40.7336,
        "lon": -74.0027
    },

    "East Village": {
        "id": 79,
        "lat": 40.7265,
        "lon": -73.9815
    },

    "Williamsburg": {
        "id": 260,
        "lat": 40.7081,
        "lon": -73.9571
    },

    "Downtown Brooklyn": {
        "id": 14,
        "lat": 40.6928,
        "lon": -73.9903
    },

    "Long Island City": {
        "id": 129,
        "lat": 40.7447,
        "lon": -73.9485
    },

    "Astoria": {
        "id": 7,
        "lat": 40.7644,
        "lon": -73.9235
    },

    "Flushing": {
        "id": 92,
        "lat": 40.7675,
        "lon": -73.8331
    },

    "Bronx Park": {
        "id": 69,
        "lat": 40.8506,
        "lon": -73.8770
    },

    "Fordham": {
        "id": 119,
        "lat": 40.8620,
        "lon": -73.8900
    }
}


location_names = list(
    locations.keys()
)


# =========================================================
# DISTANCE CALCULATION
# =========================================================

def calculate_distance(
    lat1,
    lon1,
    lat2,
    lon2
):

    """
    Calculate straight-line distance
    using the Haversine formula.

    Result is in miles.
    """

    earth_radius = 3958.8

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1)
        *
        np.cos(lat2)
        *
        np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a)
    )

    distance = earth_radius * c

    return distance


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">'
    '🚕 NYC Taxi Trip Predictor'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Enter your trip details to estimate taxi travel time.'
    '</div>',
    unsafe_allow_html=True
)


st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "🚕 NYC Taxi"
)

st.sidebar.success(
    "Machine Learning Prediction"
)

st.sidebar.write(
    "### Model"
)

st.sidebar.write(
    "XGBoost Regressor"
)

st.sidebar.write(
    "### Prediction"
)

st.sidebar.write(
    "Estimated Trip Duration"
)

st.sidebar.write(
    "### Unit"
)

st.sidebar.write(
    "Minutes"
)


# =========================================================
# LOCATION SECTION
# =========================================================

st.subheader(
    "📍 Where are you travelling?"
)


col1, col2 = st.columns(2)


with col1:

    pickup_name = st.selectbox(
        "🟢 Pickup Location",
        location_names,
        index=2,
        help="Choose where your taxi trip starts."
    )


with col2:

    dropoff_name = st.selectbox(
        "🔴 Drop-off Location",
        location_names,
        index=0,
        help="Choose where your taxi trip ends."
    )


pickup_info = locations[
    pickup_name
]

dropoff_info = locations[
    dropoff_name
]


pickup_location = pickup_info[
    "id"
]

dropoff_location = dropoff_info[
    "id"
]


# =========================================================
# AUTOMATIC DISTANCE
# =========================================================

trip_distance = calculate_distance(

    pickup_info["lat"],
    pickup_info["lon"],

    dropoff_info["lat"],
    dropoff_info["lon"]

)


# =========================================================
# MAP
# =========================================================

st.subheader(
    "🗺️ Your Route"
)


map_center = [

    (
        pickup_info["lat"]
        +
        dropoff_info["lat"]
    ) / 2,

    (
        pickup_info["lon"]
        +
        dropoff_info["lon"]
    ) / 2

]


taxi_map = folium.Map(

    location=map_center,

    zoom_start=11,

    tiles="OpenStreetMap"

)


# =========================================================
# PICKUP MARKER
# =========================================================

folium.Marker(

    location=[

        pickup_info["lat"],
        pickup_info["lon"]

    ],

    popup=(
        f"<b>Pickup</b><br>"
        f"{pickup_name}"
    ),

    tooltip="🟢 Pickup",

    icon=folium.Icon(

        color="green",

        icon="play"

    )

).add_to(taxi_map)


# =========================================================
# DROP-OFF MARKER
# =========================================================

folium.Marker(

    location=[

        dropoff_info["lat"],
        dropoff_info["lon"]

    ],

    popup=(
        f"<b>Drop-off</b><br>"
        f"{dropoff_name}"
    ),

    tooltip="🔴 Drop-off",

    icon=folium.Icon(

        color="red",

        icon="stop"

    )

).add_to(taxi_map)


# =========================================================
# ROUTE LINE
# =========================================================

folium.PolyLine(

    locations=[

        [

            pickup_info["lat"],
            pickup_info["lon"]

        ],

        [

            dropoff_info["lat"],
            dropoff_info["lon"]

        ]

    ],

    weight=5,

    opacity=0.8

).add_to(taxi_map)


st_folium(

    taxi_map,

    width=None,

    height=430

)


# =========================================================
# AUTOMATIC DISTANCE DISPLAY
# =========================================================

st.info(

    f"📏 Estimated distance between "
    f"{pickup_name} and {dropoff_name}: "
    f"**{trip_distance:.2f} miles**"

)


# =========================================================
# TRIP INFORMATION
# =========================================================

st.subheader(
    "🚕 Trip Information"
)


col1, col2 = st.columns(2)


with col1:

    passenger_count = st.number_input(

        "👥 Number of Passengers",

        min_value=1,

        max_value=8,

        value=2,

        step=1,

        help="How many passengers will travel?"

    )


with col2:

    st.metric(

        "📏 Trip Distance",

        f"{trip_distance:.2f} miles"

    )


# =========================================================
# PAYMENT METHOD
# =========================================================

st.subheader(
    "💳 Payment Method"
)


payment_options = {

    "Credit Card": 1,

    "Cash": 2,

    "No Charge": 3,

    "Dispute": 4,

    "Unknown": 5,

    "Voided Trip": 6

}


payment_name = st.selectbox(

    "How will you pay?",

    list(
        payment_options.keys()
    ),

    help=(
        "Choose your expected payment method."
    )

)


payment_type = payment_options[
    payment_name
]


# =========================================================
# TAXI FARE TYPE
# =========================================================

st.subheader(
    "💰 Taxi Fare Type"
)


fare_options = {

    "Standard City Fare": 1,

    "JFK Airport Fare": 2,

    "Newark Airport Fare": 3,

    "Nassau / Westchester Fare": 4,

    "Negotiated Fare": 5,

    "Group Ride": 6

}


fare_name = st.selectbox(

    "Select your fare type",

    list(
        fare_options.keys()
    ),

    help=(
        "Choose the fare category for your trip."
    )

)


rate_code = fare_options[
    fare_name
]


# =========================================================
# DATE & TIME
# =========================================================

st.subheader(
    "📅 When are you travelling?"
)


col1, col2 = st.columns(2)


with col1:

    pickup_date = st.date_input(

        "📅 Travel Date",

        help=(
            "Select your travel date."
        )

    )


with col2:

    pickup_time = st.time_input(

        "🕐 Pickup Time",

        help=(
            "Select approximately when "
            "your trip starts."
        )

    )


# =========================================================
# FEATURE ENGINEERING
# =========================================================

pickup_datetime = pd.Timestamp.combine(

    pickup_date,

    pickup_time

)


pickup_hour = (
    pickup_datetime.hour
)

pickup_day = (
    pickup_datetime.day
)

pickup_month = (
    pickup_datetime.month
)

pickup_weekday = (
    pickup_datetime.weekday()
)


is_weekend = int(

    pickup_weekday >= 5

)


is_rush_hour = int(

    pickup_hour in [

        7,
        8,
        9,
        16,
        17,
        18,
        19

    ]

)


distance_squared = (

    trip_distance ** 2

)


distance_per_passenger = (

    trip_distance
    /
    passenger_count

)


# =========================================================
# PREDICT BUTTON
# =========================================================

st.divider()


predict_button = st.button(

    "🚕 ESTIMATE TRIP TIME",

    type="primary",

    use_container_width=True

)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:


    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    if pickup_name == dropoff_name:

        st.warning(

            "⚠️ Pickup and drop-off locations "
            "are the same. Please choose different locations."

        )

        st.stop()


    if trip_distance <= 0:

        st.error(

            "❌ Unable to calculate the trip distance."

        )

        st.stop()


    # -----------------------------------------------------
    # CREATE INPUT DATA
    # -----------------------------------------------------

    input_data = pd.DataFrame({

        "passenger_count": [

            passenger_count

        ],

        "trip_distance": [

            trip_distance

        ],

        "RatecodeID": [

            rate_code

        ],

        "PULocationID": [

            pickup_location

        ],

        "DOLocationID": [

            dropoff_location

        ],

        "payment_type": [

            payment_type

        ],

        "pickup_hour": [

            pickup_hour

        ],

        "pickup_day": [

            pickup_day

        ],

        "pickup_month": [

            pickup_month

        ],

        "pickup_weekday": [

            pickup_weekday

        ],

        "is_weekend": [

            is_weekend

        ],

        "is_rush_hour": [

            is_rush_hour

        ],

        "distance_squared": [

            distance_squared

        ],

        "distance_per_passenger": [

            distance_per_passenger

        ]

    })


    # -----------------------------------------------------
    # CHECK FEATURES
    # -----------------------------------------------------

    missing_features = [

        feature

        for feature in feature_columns

        if feature not in input_data.columns

    ]


    if missing_features:

        st.error(
            "❌ Model feature mismatch."
        )

        st.write(
            "The following features are required "
            "by your trained model but are missing:"
        )

        st.code(
            "\n".join(
                missing_features
            )
        )

        st.write(
            "Your model expects:"
        )

        st.write(
            feature_columns
        )

        st.stop()


    # -----------------------------------------------------
    # EXACT FEATURE ORDER
    # -----------------------------------------------------

    input_data = input_data[

        feature_columns

    ]


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    try:

        prediction = model.predict(

            input_data

        )[0]

    except Exception as error:

        st.error(
            "❌ Prediction failed."
        )

        st.exception(
            error
        )

        st.stop()


    # -----------------------------------------------------
    # CLEAN PREDICTION
    # -----------------------------------------------------

    prediction = float(
        prediction
    )


    prediction = max(
        1,
        prediction
    )


    prediction = min(
        180,
        prediction
    )


    # =====================================================
    # ESTIMATED SPEED
    # =====================================================

    estimated_speed = (

        trip_distance
        /
        (prediction / 60)

    )


    # =====================================================
    # RESULT
    # =====================================================

    st.divider()


    st.subheader(
        "🎯 Your Trip Estimate"
    )


    result1, result2, result3 = st.columns(3)


    with result1:

        st.metric(

            "⏱️ Estimated Time",

            f"{prediction:.1f} min"

        )


    with result2:

        st.metric(

            "📏 Distance",

            f"{trip_distance:.2f} miles"

        )


    with result3:

        st.metric(

            "🚕 Average Speed",

            f"{estimated_speed:.1f} mph"

        )


    # =====================================================
    # FRIENDLY MESSAGE
    # =====================================================

    if prediction < 10:

        st.success(

            f"🚕 Your trip should take "
            f"approximately **{prediction:.1f} minutes**."

        )

    elif prediction < 30:

        st.info(

            f"🚕 Your trip should take "
            f"approximately **{prediction:.1f} minutes**."

        )

    else:

        st.warning(

            f"🚕 Your trip may take around "
            f"**{prediction:.1f} minutes**."

        )


    # =====================================================
    # TRIP SUMMARY
    # =====================================================

    st.subheader(
        "📋 Trip Summary"
    )


    summary = pd.DataFrame({

        "Trip Information": [

            "🟢 Pickup",

            "🔴 Drop-off",

            "👥 Passengers",

            "📏 Distance",

            "💳 Payment",

            "💰 Fare Type",

            "📅 Date",

            "🕐 Time"

        ],

        "Details": [

            pickup_name,

            dropoff_name,

            passenger_count,

            f"{trip_distance:.2f} miles",

            payment_name,

            fare_name,

            pickup_date.strftime(
                "%d-%m-%Y"
            ),

            pickup_time.strftime(
                "%I:%M %p"
            )

        ]

    })


    st.dataframe(

        summary,

        use_container_width=True,

        hide_index=True

    )


    # =====================================================
    # TRIP CONDITIONS
    # =====================================================

    st.subheader(
        "🚦 Trip Conditions"
    )


    condition1, condition2 = st.columns(2)


    with condition1:

        if is_weekend:

            st.info(
                "📅 Weekend trip"
            )

        else:

            st.info(
                "📅 Weekday trip"
            )


    with condition2:

        if is_rush_hour:

            st.warning(
                "🚦 Rush-hour period"
            )

        else:

            st.success(
                "🚦 Outside rush hour"
            )


    # =====================================================
    # TECHNICAL DETAILS
    # =====================================================

    with st.expander(
        "🔧 Technical Model Details"
    ):

        st.write(
            "**Algorithm:** XGBoost Regressor"
        )

        st.write(
            "**Target:** Trip Duration"
        )

        st.write(
            "**Prediction:** Minutes"
        )

        st.write(
            "**Features used:**",
            len(feature_columns)
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🚕 NYC Taxi Trip Predictor | "
    "Machine Learning Project"
)
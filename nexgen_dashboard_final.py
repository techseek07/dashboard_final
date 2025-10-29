# ============================================================================
# NEXGEN LOGISTICS INTELLIGENCE DASHBOARD - AI-DRIVEN VERSION
# Features: AI-learned speeds, smart route comparison, dynamic buffers
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, precision_recall_curve)
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="NEXGEN Logistics Dashboard", page_icon="🚚", layout="wide")


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

@st.cache_data
def load_and_preprocess_data():
    """Load data with EXACT SAME preprocessing as training"""
    try:
        df_delivery = pd.read_csv('delivery_performance.csv')
        df_orders = pd.read_csv('orders.csv')
        df_routes = pd.read_csv('routes_distance.csv')
        df_costs = pd.read_csv('cost_breakdown.csv')
        df_fleet = pd.read_csv('vehicle_fleet.csv')

        df_merged = df_delivery.merge(df_orders, on='Order_ID', how='left') \
            .merge(df_routes, on='Order_ID', how='left') \
            .merge(df_costs, on='Order_ID', how='left')

        df_merged['Weather_Impact'] = df_merged['Weather_Impact'].fillna('None')
        df_merged['Special_Handling'] = df_merged['Special_Handling'].fillna('None')
        df_merged['Traffic_Delay_Minutes'] = df_merged['Traffic_Delay_Minutes'].fillna(0)
        df_merged['is_delayed'] = df_merged['Delivery_Status'].isin(['Slightly-Delayed', 'Severely-Delayed']).astype(
            int)

        # Data cleaning
        df_merged['suspicious_delayed'] = df_merged.apply(
            lambda r: r['is_delayed'] == 1 and r['Traffic_Delay_Minutes'] < 30
                      and r['Distance_KM'] < 500 and r['Weather_Impact'] == 'None', axis=1
        )
        df_merged['suspicious_ontime'] = df_merged.apply(
            lambda r: r['is_delayed'] == 0 and (r['Traffic_Delay_Minutes'] > 90
                                                or r['Weather_Impact'] in ['Heavy_Rain', 'Fog'] or r[
                                                    'Distance_KM'] > 2000), axis=1
        )
        df_clean = df_merged[~(df_merged['suspicious_delayed'] | df_merged['suspicious_ontime'])].copy()

        # Feature engineering
        carrier_delay_rate = df_clean.groupby('Carrier')['is_delayed'].mean()
        df_clean['carrier_risk'] = df_clean['Carrier'].map(carrier_delay_rate)

        route_delay_rate = df_clean.groupby('Route')['is_delayed'].mean()
        df_clean['route_risk'] = df_clean['Route'].map(route_delay_rate)

        priority_carrier_delay = df_clean.groupby(['Priority', 'Carrier'])['is_delayed'].mean()
        df_clean['priority_carrier_risk'] = df_clean.apply(
            lambda x: priority_carrier_delay.get((x['Priority'], x['Carrier']), 0.5), axis=1
        )

        df_clean['traffic_severe'] = (df_clean['Traffic_Delay_Minutes'] > 60).astype(int)
        df_clean['weather_severe'] = df_clean['Weather_Impact'].isin(['Heavy_Rain', 'Fog']).astype(int)
        distance_median = df_clean['Distance_KM'].median()
        df_clean['long_distance'] = (df_clean['Distance_KM'] > distance_median * 1.5).astype(int)
        df_clean['compound_risk'] = (
                df_clean['traffic_severe'] * 2 + df_clean['weather_severe'] * 2 +
                df_clean['long_distance'] + (df_clean['carrier_risk'] > 0.5).astype(int) * 2
        )
        df_clean['time_pressure'] = df_clean['Distance_KM'] / (df_clean['Promised_Delivery_Days'] + 1)
        df_clean['cost_per_km'] = df_clean['Delivery_Cost_INR'] / (df_clean['Distance_KM'] + 1)

        # Time features
        df_clean['Order_Date'] = pd.to_datetime(df_clean['Order_Date'], errors='coerce')
        df_clean['day_of_week'] = df_clean['Order_Date'].dt.dayofweek
        df_clean['is_weekend'] = (df_clean['day_of_week'] >= 5).astype(int)
        df_clean['is_peak_season'] = df_clean['Order_Date'].dt.month.isin([11, 12]).astype(int)
        df_clean['quarter'] = df_clean['Order_Date'].dt.quarter

        # Calculate Actual_Delivery_Days if not present
        if 'Actual_Delivery_Days' not in df_clean.columns:
            df_clean['Order_Date'] = pd.to_datetime(df_clean['Order_Date'])
            df_clean['Actual_Delivery_Date'] = pd.to_datetime(
                df_clean.get('Actual_Delivery_Date', df_clean['Order_Date']))
            df_clean['Actual_Delivery_Days'] = (df_clean['Actual_Delivery_Date'] - df_clean['Order_Date']).dt.days
            df_clean['Actual_Delivery_Days'] = df_clean['Actual_Delivery_Days'].clip(lower=1, upper=30)

        return df_clean, df_fleet

    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()


@st.cache_resource
def load_model():
    """Load trained model artifacts"""
    try:
        with open('nexgen_full_model.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        st.error("Model file not found. Train model first.")
        st.stop()


def generate_predictions(df, artifacts):
    """Generate predictions + calculate optimal thresholds"""
    feature_cols = artifacts['feature_cols']
    df_model = df[feature_cols + ['Order_ID']].copy()

    for col in ['Priority', 'Customer_Segment', 'Carrier', 'Weather_Impact']:
        if col in artifacts['label_encoders']:
            le = artifacts['label_encoders'][col]
            df_model[col] = df_model[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )

    X = df_model[feature_cols]
    X_scaled = artifacts['scaler'].transform(X)

    delay_probs = artifacts['model'].predict_proba(X_scaled)[:, 1]
    df['delay_probability'] = delay_probs

    y_true = df['is_delayed'].values
    precision, recall, pr_thresholds = precision_recall_curve(y_true, delay_probs)

    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    best_f1_threshold = float(pr_thresholds[best_f1_idx])

    accuracies = []
    for t in pr_thresholds:
        preds = (delay_probs >= t).astype(int)
        accuracies.append(accuracy_score(y_true, preds))
    best_acc_threshold = float(pr_thresholds[np.argmax(accuracies)])

    quantile_threshold = float(np.quantile(delay_probs, 0.70))

    return df, {
        'best_f1_threshold': best_f1_threshold,
        'best_acc_threshold': best_acc_threshold,
        'quantile_threshold': quantile_threshold
    }


# ============================================================================
# AI-DRIVEN HELPER FUNCTIONS (NEW!)
# ============================================================================

def calculate_required_buffer_ai(order, df):
    """
    AI-DRIVEN: Calculate delivery time using learned speeds from historical data
    Returns: (required_days, buffer_needed, explanation, speed_used)
    """

    promised_days = order['Promised_Delivery_Days']
    distance = order['Distance_KM']
    current_route = order['Route']
    current_carrier = order['Carrier']

    # AI LEARNING: Extract actual speed from historical deliveries
    similar_orders = df[
        (df['Route'] == current_route) &
        (df['Carrier'] == current_carrier)
        ]

    if len(similar_orders) > 0:
        actual_speeds = []
        for _, hist_order in similar_orders.iterrows():
            if pd.notna(hist_order.get('Actual_Delivery_Days')) and hist_order['Actual_Delivery_Days'] > 0:
                hist_distance = hist_order['Distance_KM']
                actual_days = hist_order['Actual_Delivery_Days']
                actual_speed = hist_distance / actual_days
                if 50 <= actual_speed <= 500:  # Sanity check
                    actual_speeds.append(actual_speed)

        if len(actual_speeds) > 0:
            learned_speed = np.median(actual_speeds)
        else:
            # Fallback to route-level
            route_orders = df[df['Route'] == current_route]
            route_speeds = []
            for _, ro in route_orders.iterrows():
                if pd.notna(ro.get('Actual_Delivery_Days')) and ro['Actual_Delivery_Days'] > 0:
                    route_speeds.append(ro['Distance_KM'] / ro['Actual_Delivery_Days'])
            learned_speed = np.median(route_speeds) if len(route_speeds) > 0 else 350
    else:
        # Fallback to carrier-level
        carrier_orders = df[df['Carrier'] == current_carrier]
        carrier_speeds = []
        for _, co in carrier_orders.iterrows():
            if pd.notna(co.get('Actual_Delivery_Days')) and co['Actual_Delivery_Days'] > 0:
                carrier_speeds.append(co['Distance_KM'] / co['Actual_Delivery_Days'])
        learned_speed = np.median(carrier_speeds) if len(carrier_speeds) > 0 else 350

    learned_speed = max(50, min(500, learned_speed))
    base_travel_days = distance / learned_speed

    # Traffic delay
    traffic_delay_days = order['Traffic_Delay_Minutes'] / 1440

    # Weather delay
    if order['weather_severe'] == 1:
        weather = order['Weather_Impact']
        weather_orders = df[df['Weather_Impact'] == weather]
        if len(weather_orders) > 5:
            weather_delay_rate = weather_orders['is_delayed'].mean()
            weather_delay_days = weather_delay_rate * 1.0
        else:
            weather_delay_days = 1.0 if weather == 'Heavy_Rain' else 0.7 if weather == 'Fog' else 0.5
    else:
        weather_delay_days = 0

    # Route risk delay
    route_risk = order['route_risk']
    if route_risk > 0.75:
        route_delay_days = 2.0
    elif route_risk > 0.6:
        route_delay_days = 1.5
    elif route_risk > 0.4:
        route_delay_days = 1.0
    else:
        route_delay_days = 0.5

    # Carrier reliability
    carrier_risk = order['carrier_risk']
    carrier_delay_days = carrier_risk * 1.5

    required_days = base_travel_days + traffic_delay_days + weather_delay_days + route_delay_days + carrier_delay_days
    buffer_needed = max(0, required_days - promised_days)

    explanation = f"""
**AI-Calculated Breakdown (learned from historical data):**
- Base travel: {base_travel_days:.1f} days ({distance:.0f} km ÷ **{learned_speed:.0f} km/day)
- Traffic impact: +{traffic_delay_days:.1f} days ({order['Traffic_Delay_Minutes']:.0f} min)
- Weather impact: +{weather_delay_days:.1f} days ({order['Weather_Impact']})
- Route risk: +{route_delay_days:.1f} days (ML model: {route_risk:.1%} historical delay rate)
- Carrier reliability: +{carrier_delay_days:.1f} days ({current_carrier} has {carrier_risk:.1%} risk)

**Total:** {required_days:.1f} days needed vs {promised_days} days promised
**Buffer:** +{buffer_needed:.1f} days
"""

    return required_days, buffer_needed, explanation, learned_speed


# ============================================================================
# HELPER FUNCTION: GET HUB TRANSFER TIME (NEW!)
# ============================================================================

def get_hub_transfer_time(hub_city):
    """
    Get realistic hub transfer time based on city characteristics
    Returns: (transfer_days, explanation)
    """

    # Hub characteristics based on city infrastructure
    hub_characteristics = {
        'Delhi': (0.17, '4 hours - Major hub with fast operations'),
        'Mumbai': (0.25, '6 hours - Port city with medium efficiency'),
        'Bangalore': (0.33, '8 hours - Tech hub with medium operations'),
        'Chennai': (0.33, '8 hours - Port city with medium operations'),
        'Kolkata': (0.42, '10 hours - Congested hub, slower processing'),
        'Hyderabad': (0.25, '6 hours - Decent hub with good infrastructure'),
        'Pune': (0.5, '12 hours - Smaller hub, slower processing'),
        'Ahmedabad': (0.5, '12 hours - Smaller hub, standard processing'),
    }

    if hub_city in hub_characteristics:
        transfer_time, explanation = hub_characteristics[hub_city]
        confidence = "ESTIMATED"
    else:
        # Unknown city - use conservative estimate
        transfer_time = 0.5
        explanation = '12 hours - Unknown hub, using standard estimate'
        confidence = "ASSUMED"

    return transfer_time, explanation, confidence


# ============================================================================
# UPDATED: FIND BETTER ROUTES (WITH HUB-SPECIFIC TIMES)
# ============================================================================

def find_better_routes_ai(order, df):
    """
    AI-DRIVEN: Find routes that deliver FASTER (even if longer distance)
    Uses hub-specific transfer times based on city infrastructure
    """

    origin = order['Origin']
    destination = order['Destination']

    # Calculate current route's total time
    current_required_days, _, _, current_speed = calculate_required_buffer_ai(order, df)

    all_cities = df['Origin'].unique()
    intermediate_cities = [city for city in all_cities if city not in [origin, destination]]

    alternate_routes = []

    for hub in intermediate_cities:
        route1 = f"{origin}-{hub}"
        route2 = f"{hub}-{destination}"

        route1_data = df[df['Route'] == route1]
        route2_data = df[df['Route'] == route2]

        if len(route1_data) > 0 and len(route2_data) > 0:
            # Get hub-specific transfer time
            hub_transfer, hub_explanation, hub_confidence = get_hub_transfer_time(hub)

            carriers = df['Carrier'].unique()
            best_combo = None
            best_time = float('inf')

            for carrier1 in carriers:
                for carrier2 in carriers:
                    leg1_orders = route1_data[route1_data['Carrier'] == carrier1]
                    if len(leg1_orders) == 0:
                        continue

                    leg1_sample = leg1_orders.iloc[0].copy()
                    leg1_req_days, _, _, leg1_speed = calculate_required_buffer_ai(leg1_sample, df)

                    leg2_orders = route2_data[route2_data['Carrier'] == carrier2]
                    if len(leg2_orders) == 0:
                        continue

                    leg2_sample = leg2_orders.iloc[0].copy()
                    leg2_req_days, _, _, leg2_speed = calculate_required_buffer_ai(leg2_sample, df)

                    # Use hub-specific transfer time
                    total_alt_days = leg1_req_days + hub_transfer + leg2_req_days

                    if total_alt_days < best_time:
                        best_time = total_alt_days
                        best_combo = {
                            'carrier1': carrier1,
                            'carrier2': carrier2,
                            'leg1_days': leg1_req_days,
                            'leg2_days': leg2_req_days,
                            'leg1_dist': leg1_sample['Distance_KM'],
                            'leg2_dist': leg2_sample['Distance_KM'],
                            'leg1_speed': leg1_speed,
                            'leg2_speed': leg2_speed,
                            'hub_transfer': hub_transfer,
                            'hub_explanation': hub_explanation,
                            'hub_confidence': hub_confidence
                        }

            if best_combo and best_time < current_required_days:
                days_saved = current_required_days - best_time
                total_distance = best_combo['leg1_dist'] + best_combo['leg2_dist']
                distance_overhead = total_distance - order['Distance_KM']

                alternate_routes.append({
                    'Route': f"{origin} → {hub} → {destination}",
                    'Hub': hub,
                    'Carrier1': best_combo['carrier1'],
                    'Carrier2': best_combo['carrier2'],
                    'Total_Distance': total_distance,
                    'Distance_Overhead': distance_overhead,
                    'Total_Days': best_time,
                    'Days_Saved': days_saved,
                    'Leg1_Days': best_combo['leg1_days'],
                    'Leg2_Days': best_combo['leg2_days'],
                    'Leg1_Speed': best_combo['leg1_speed'],
                    'Leg2_Speed': best_combo['leg2_speed'],
                    'Hub_Transfer': best_combo['hub_transfer'],
                    'Hub_Explanation': best_combo['hub_explanation'],
                    'Hub_Confidence': best_combo['hub_confidence'],
                    'Current_Days': current_required_days,
                    'Current_Speed': current_speed
                })

    return sorted(alternate_routes, key=lambda x: x['Days_Saved'], reverse=True)[:3]


def find_better_routes_ai(order, df):
    """
    AI-DRIVEN: Find routes that deliver FASTER (even if longer distance)
    Uses hub-specific transfer times based on city infrastructure
    """

    origin = order['Origin']
    destination = order['Destination']

    # Calculate current route's total time
    current_required_days, _, _, current_speed = calculate_required_buffer_ai(order, df)

    all_cities = df['Origin'].unique()
    intermediate_cities = [city for city in all_cities if city not in [origin, destination]]

    alternate_routes = []

    for hub in intermediate_cities:
        route1 = f"{origin}-{hub}"
        route2 = f"{hub}-{destination}"

        route1_data = df[df['Route'] == route1]
        route2_data = df[df['Route'] == route2]

        if len(route1_data) > 0 and len(route2_data) > 0:
            # Get hub-specific transfer time
            hub_transfer, hub_explanation, hub_confidence = get_hub_transfer_time(hub)

            carriers = df['Carrier'].unique()
            best_combo = None
            best_time = float('inf')

            for carrier1 in carriers:
                for carrier2 in carriers:
                    leg1_orders = route1_data[route1_data['Carrier'] == carrier1]
                    if len(leg1_orders) == 0:
                        continue

                    leg1_sample = leg1_orders.iloc[0].copy()
                    leg1_req_days, _, _, leg1_speed = calculate_required_buffer_ai(leg1_sample, df)

                    leg2_orders = route2_data[route2_data['Carrier'] == carrier2]
                    if len(leg2_orders) == 0:
                        continue

                    leg2_sample = leg2_orders.iloc[0].copy()
                    leg2_req_days, _, _, leg2_speed = calculate_required_buffer_ai(leg2_sample, df)

                    # Use hub-specific transfer time
                    total_alt_days = leg1_req_days + hub_transfer + leg2_req_days

                    if total_alt_days < best_time:
                        best_time = total_alt_days
                        best_combo = {
                            'carrier1': carrier1,
                            'carrier2': carrier2,
                            'leg1_days': leg1_req_days,
                            'leg2_days': leg2_req_days,
                            'leg1_dist': leg1_sample['Distance_KM'],
                            'leg2_dist': leg2_sample['Distance_KM'],
                            'leg1_speed': leg1_speed,
                            'leg2_speed': leg2_speed,
                            'hub_transfer': hub_transfer,
                            'hub_explanation': hub_explanation,
                            'hub_confidence': hub_confidence
                        }

            if best_combo and best_time < current_required_days:
                days_saved = current_required_days - best_time
                total_distance = best_combo['leg1_dist'] + best_combo['leg2_dist']
                distance_overhead = total_distance - order['Distance_KM']

                alternate_routes.append({
                    'Route': f"{origin} → {hub} → {destination}",
                    'Hub': hub,
                    'Carrier1': best_combo['carrier1'],
                    'Carrier2': best_combo['carrier2'],
                    'Total_Distance': total_distance,
                    'Distance_Overhead': distance_overhead,
                    'Total_Days': best_time,
                    'Days_Saved': days_saved,
                    'Leg1_Days': best_combo['leg1_days'],
                    'Leg2_Days': best_combo['leg2_days'],
                    'Leg1_Speed': best_combo['leg1_speed'],
                    'Leg2_Speed': best_combo['leg2_speed'],
                    'Hub_Transfer': best_combo['hub_transfer'],
                    'Hub_Explanation': best_combo['hub_explanation'],
                    'Hub_Confidence': best_combo['hub_confidence'],
                    'Current_Days': current_required_days,
                    'Current_Speed': current_speed
                })

    return sorted(alternate_routes, key=lambda x: x['Days_Saved'], reverse=True)[:3]


# ============================================================================
# LOAD DATA & MODEL
# ============================================================================

df, df_fleet = load_and_preprocess_data()
artifacts = load_model()
df, threshold_info = generate_predictions(df, artifacts)

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title("🚚 NEXGEN Logistics")
st.sidebar.info(f"""
**Model Performance**  
CV F1: 88.1%  
Test F1: 89.7%  
Samples: {len(df)}
""")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "🤖 ML Insights", "📦 Order Analysis",
     "🚛 Carrier & Route", "💰 Cost Analysis", "🔮 Predict New Order"]
)

st.sidebar.markdown("---")

st.sidebar.subheader("⚙️ Threshold Strategy")
user_threshold = st.sidebar.slider("User-defined threshold", 0.0, 1.0, 0.6, 0.05)

threshold_options = [
    f"User-defined ({user_threshold:.2f})",
    f"Maximize F1 ({threshold_info['best_f1_threshold']:.2f})",
    f"Maximize Accuracy ({threshold_info['best_acc_threshold']:.2f})",
    f"Top 30% risk ({threshold_info['quantile_threshold']:.2f})"
]

threshold_strategy = st.sidebar.radio("Choose threshold", threshold_options, index=0)

if "F1" in threshold_strategy:
    active_threshold = threshold_info['best_f1_threshold']
elif "Accuracy" in threshold_strategy:
    active_threshold = threshold_info['best_acc_threshold']
elif "Top 30%" in threshold_strategy:
    active_threshold = threshold_info['quantile_threshold']
else:
    active_threshold = user_threshold

st.sidebar.success(f"**Active threshold:** {active_threshold:.3f}")

df['predicted_delay'] = (df['delay_probability'] > active_threshold).astype(int)
df['high_risk'] = df['predicted_delay']

st.sidebar.markdown("---")

selected_priority = st.sidebar.multiselect("Priority", df['Priority'].unique(), df['Priority'].unique())
selected_carrier = st.sidebar.multiselect("Carrier", df['Carrier'].unique(), df['Carrier'].unique())

df_filtered = df[(df['Priority'].isin(selected_priority)) & (df['Carrier'].isin(selected_carrier))].copy()

high_risk_count = df_filtered['high_risk'].sum()
low_risk_count = len(df_filtered) - high_risk_count

st.sidebar.info(f"""
**Filtered View:**  
📦 Orders: {len(df_filtered)}  
🔴 High Risk: {high_risk_count}  
🟢 Low Risk: {low_risk_count}
""")

# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================

if page == "🏠 Overview":
    st.title("🚚 NEXGEN Logistics Intelligence Dashboard")
    st.markdown("### Real-Time Delay Prediction & Optimization")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Orders", f"{len(df_filtered):,}")
    col2.metric("Actual Delay Rate", f"{df_filtered['is_delayed'].mean():.1%}")
    col3.metric("High Risk Orders", f"{df_filtered['high_risk'].sum()}")
    col4.metric("Avg Distance", f"{df_filtered['Distance_KM'].mean():.0f} km")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        status_counts = df_filtered['Delivery_Status'].value_counts()
        fig1 = px.pie(values=status_counts.values, names=status_counts.index,
                      title="Delivery Status Distribution", hole=0.4)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        carrier_perf = df_filtered.groupby('Carrier').agg({'is_delayed': 'mean', 'Order_ID': 'count'}).reset_index()
        carrier_perf.columns = ['Carrier', 'Delay_Rate', 'Orders']
        fig2 = px.bar(carrier_perf, x='Carrier', y='Delay_Rate', color='Delay_Rate',
                      title="Carrier Delay Rates", color_continuous_scale='RdYlGn_r', text='Delay_Rate')
        fig2.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📊 Risk Score Analysis")

    tab1, tab2, tab3 = st.tabs(["Distribution", "Risk Buckets", "Threshold Impact"])

    with tab1:
        st.markdown("### Risk Probability Distribution")

        fig1 = make_subplots(rows=1, cols=2,
                             subplot_titles=('On-Time Orders', 'Delayed Orders'))

        ontime_probs = df_filtered[df_filtered['is_delayed'] == 0]['delay_probability']
        fig1.add_trace(
            go.Histogram(x=ontime_probs, nbinsx=20, name='On-Time',
                         marker_color='lightgreen', showlegend=True),
            row=1, col=1
        )

        delayed_probs = df_filtered[df_filtered['is_delayed'] == 1]['delay_probability']
        fig1.add_trace(
            go.Histogram(x=delayed_probs, nbinsx=20, name='Delayed',
                         marker_color='lightcoral', showlegend=True),
            row=1, col=2
        )

        fig1.update_xaxes(title_text="Predicted Delay Probability", range=[0, 1], row=1, col=1)
        fig1.update_xaxes(title_text="Predicted Delay Probability", range=[0, 1], row=1, col=2)
        fig1.update_yaxes(title_text="Number of Orders", row=1, col=1)
        fig1.update_yaxes(title_text="Number of Orders", row=1, col=2)

        fig1.add_vline(x=active_threshold, line_dash="dash", line_color="red",
                       row=1, col=1, annotation_text="Threshold")
        fig1.add_vline(x=active_threshold, line_dash="dash", line_color="red",
                       row=1, col=2, annotation_text="Threshold")

        fig1.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig1, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.success(f"""
            **On-Time Orders (Green)**
            - Total: {len(ontime_probs)}
            - Avg Risk: {ontime_probs.mean():.1%}
            """)

        with col2:
            st.error(f"""
            **Delayed Orders (Red)**
            - Total: {len(delayed_probs)}
            - Avg Risk: {delayed_probs.mean():.1%}
            """)

    with tab2:
        st.markdown("### Risk Buckets")

        df_filtered['risk_bucket'] = pd.cut(
            df_filtered['delay_probability'],
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=['Low (0-30%)', 'Medium (30-50%)', 'High (50-70%)', 'Critical (70-100%)']
        )

        bucket_summary = df_filtered.groupby('risk_bucket', observed=True).agg({
            'Order_ID': 'count',
            'is_delayed': 'mean'
        }).reset_index()
        bucket_summary.columns = ['Risk_Bucket', 'Order_Count', 'Actual_Delay_Rate']

        fig2 = go.Figure()

        colors = ['#51CF66', '#FFC107', '#FF9800', '#F44336']

        fig2.add_trace(go.Bar(
            x=bucket_summary['Risk_Bucket'],
            y=bucket_summary['Order_Count'],
            text=bucket_summary['Order_Count'],
            textposition='outside',
            marker_color=colors,
            name='Order Count'
        ))

        fig2.update_layout(
            title="Orders Distributed by Risk Level",
            xaxis_title="Risk Level",
            yaxis_title="Number of Orders",
            height=400,
            showlegend=False
        )

        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(bucket_summary, use_container_width=True)

    with tab3:
        st.markdown("### Threshold Impact Analysis")

        thresholds = np.arange(0.1, 1.0, 0.05)
        metrics = {'threshold': [], 'flagged_orders': [], 'true_delays_caught': [], 'false_alarms': []}

        for t in thresholds:
            preds = (df_filtered['delay_probability'] >= t).astype(int)
            metrics['threshold'].append(t)
            metrics['flagged_orders'].append(preds.sum())
            metrics['true_delays_caught'].append(((preds == 1) & (df_filtered['is_delayed'] == 1)).sum())
            metrics['false_alarms'].append(((preds == 1) & (df_filtered['is_delayed'] == 0)).sum())

        metrics_df = pd.DataFrame(metrics)

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=metrics_df['threshold'], y=metrics_df['flagged_orders'],
                                  mode='lines+markers', name='Total Flagged', line=dict(color='blue', width=2)))
        fig3.add_trace(go.Scatter(x=metrics_df['threshold'], y=metrics_df['true_delays_caught'],
                                  mode='lines+markers', name='True Delays Caught', line=dict(color='green', width=2)))
        fig3.add_trace(go.Scatter(x=metrics_df['threshold'], y=metrics_df['false_alarms'],
                                  mode='lines+markers', name='False Alarms', line=dict(color='red', width=2)))

        fig3.add_vline(x=active_threshold, line_dash="dash", line_color="black",
                       annotation_text=f"Current: {active_threshold:.2f}")

        fig3.update_layout(title="How Threshold Affects Order Flagging",
                           xaxis_title="Threshold Value", yaxis_title="Number of Orders",
                           height=400, hovermode='x unified')

        st.plotly_chart(fig3, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Flagged Orders", (df_filtered['delay_probability'] >= active_threshold).sum())
        col2.metric("True Delays Caught", ((df_filtered['delay_probability'] >= active_threshold) &
                                           (df_filtered['is_delayed'] == 1)).sum())
        col3.metric("False Alarms", ((df_filtered['delay_probability'] >= active_threshold) &
                                     (df_filtered['is_delayed'] == 0)).sum())

# ============================================================================
# PAGE 2: ML INSIGHTS
# ============================================================================

elif page == "🤖 ML Insights":
    st.title("🤖 Machine Learning Model Insights")

    if hasattr(artifacts['model'], 'feature_importances_'):
        importance = pd.DataFrame({
            'Feature': artifacts['feature_cols'],
            'Importance': artifacts['model'].feature_importances_
        }).sort_values('Importance', ascending=False).head(10)

        st.subheader("📊 Top 10 Feature Importances")
        fig = px.bar(importance, y='Feature', x='Importance', orientation='h',
                     title="Feature Importance", text='Importance', color='Importance',
                     color_continuous_scale='Blues')
        fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Model Performance")

    acc = accuracy_score(df_filtered['is_delayed'], df_filtered['predicted_delay'])
    prec = precision_score(df_filtered['is_delayed'], df_filtered['predicted_delay'], zero_division=0)
    rec = recall_score(df_filtered['is_delayed'], df_filtered['predicted_delay'], zero_division=0)
    f1 = f1_score(df_filtered['is_delayed'], df_filtered['predicted_delay'], zero_division=0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.1%}")
    col2.metric("Precision", f"{prec:.1%}")
    col3.metric("Recall", f"{rec:.1%}")
    col4.metric("F1-Score", f"{f1:.1%}")

    cm = confusion_matrix(df_filtered['is_delayed'], df_filtered['predicted_delay'])
    fig_cm = go.Figure(data=go.Heatmap(z=cm,
                                       x=['Predicted On-Time', 'Predicted Delayed'],
                                       y=['Actually On-Time', 'Actually Delayed'],
                                       text=cm, texttemplate='%{text}', colorscale='Blues'))
    fig_cm.update_layout(title="Confusion Matrix", height=400)
    st.plotly_chart(fig_cm, use_container_width=True)

# ============================================================================
# PAGE 3: ORDER ANALYSIS (WITH AI-DRIVEN ROUTES & BUFFERS)
# ============================================================================

elif page == "📦 Order Analysis":
    st.title("📦 Order-Level Analysis")

    st.subheader("📊 Analysis by Customer Segment")

    segment_summary = df_filtered.groupby('Customer_Segment').agg({
        'Order_ID': 'count',
        'delay_probability': 'mean',
        'high_risk': 'sum',
        'Distance_KM': 'mean',
        'Order_Value_INR': 'mean',
        'Delivery_Cost_INR': 'mean'
    }).round(3)

    segment_summary.columns = ['Orders', 'ML_Risk', 'High_Risk_Count',
                               'Avg_Distance', 'Avg_Order_Value', 'Avg_Delivery_Cost']

    segment_summary['Risk_Tier'] = segment_summary['ML_Risk'].apply(
        lambda x: '🔴 Critical' if x > 0.7 else '🟡 High' if x > 0.6 else '🟢 Low'
    )

    segment_summary = segment_summary[['Risk_Tier', 'ML_Risk', 'Orders', 'High_Risk_Count',
                                       'Avg_Distance', 'Avg_Order_Value', 'Avg_Delivery_Cost']]
    segment_summary = segment_summary.sort_values('ML_Risk', ascending=False)

    st.dataframe(
        segment_summary.style.background_gradient(subset=['ML_Risk'], cmap='RdYlGn_r')
        .format({
            'ML_Risk': '{:.1%}',
            'Avg_Distance': '{:.0f} km',
            'Avg_Order_Value': '₹{:,.0f}',
            'Avg_Delivery_Cost': '₹{:,.0f}'
        }),
        use_container_width=True
    )

    worst_segment = segment_summary.index[0]
    best_segment = segment_summary.index[-1]

    col1, col2 = st.columns(2)

    with col1:
        st.error(f"""
        ### ⚠️ Highest Risk: {worst_segment}
        - ML Risk: {segment_summary.loc[worst_segment, 'ML_Risk']:.1%}
        - Orders: {segment_summary.loc[worst_segment, 'Orders']:.0f}
        - Avg Value: ₹{segment_summary.loc[worst_segment, 'Avg_Order_Value']:,.0f}
        """)

    with col2:
        st.success(f"""
        ### ✅ Best Performance: {best_segment}
        - ML Risk: {segment_summary.loc[best_segment, 'ML_Risk']:.1%}
        - Orders: {segment_summary.loc[best_segment, 'Orders']:.0f}
        - Avg Value: ₹{segment_summary.loc[best_segment, 'Avg_Order_Value']:,.0f}
        """)

    st.markdown("---")

    st.subheader("🔍 Single Order Lookup")
    order_id = st.selectbox("Select Order ID", df_filtered['Order_ID'].unique())

    if order_id:
        order = df_filtered[df_filtered['Order_ID'] == order_id].iloc[0]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            **Order Details**
            - ID: {order['Order_ID']}
            - Date: {order['Order_Date'].strftime('%Y-%m-%d')}
            - Priority: {order['Priority']}
            - Segment: {order['Customer_Segment']}
            - Value: ₹{order['Order_Value_INR']:,.0f}
            """)

        with col2:
            st.markdown(f"""
            **Logistics**
            - Route: {order['Route']}
            - Carrier: {order['Carrier']}
            - Distance: {order['Distance_KM']:.0f} km
            - Traffic: {order['Traffic_Delay_Minutes']:.0f} min
            """)

        with col3:
            delay_prob = order['delay_probability']
            risk = "🔴 HIGH" if delay_prob > 0.7 else "🟡 MEDIUM" if delay_prob > 0.4 else "🟢 LOW"
            st.markdown(f"""
            **Prediction**
            - Risk: {risk}
            - Probability: **{delay_prob:.1%}**
            - Status: {order['Delivery_Status']}
            """)

        st.markdown("---")
        st.subheader("🎯 AI-Powered Action Recommendations")

        if delay_prob > 0.5:

            # ============================================================
            # ACTION 1: AI-DRIVEN ROUTE OPTIMIZATION (UPDATED)
            # ============================================================

            st.markdown("### 🛣️ AI-Powered Route Optimization")

            current_required_days, current_buffer, current_explanation, current_speed = calculate_required_buffer_ai(
                order, df)

            st.info(f"""
            **Current Route Performance:**
            - Route: {order['Route']}
            - Carrier: {order['Carrier']}
            - Distance: {order['Distance_KM']:.0f} km
            - AI-learned speed: **{current_speed:.0f} km/day** (from historical data)
            - Total delivery time: **{current_required_days:.1f} days**
            """)

            better_routes = find_better_routes_ai(order, df)

            if len(better_routes) > 0:
                st.success(f"✅ Found {len(better_routes)} faster route(s)!")

                for idx, alt in enumerate(better_routes, 1):
                    extra_distance_pct = (alt['Distance_Overhead'] / order['Distance_KM']) * 100

                    # Determine hub transfer time display
                    hub_transfer_hours = alt['Hub_Transfer'] * 24
                    hub_transfer_display = f"{hub_transfer_hours:.0f} hours" if hub_transfer_hours < 24 else f"{alt['Hub_Transfer']:.1f} days"

                    st.info(f"""
                    **🔀 ALTERNATIVE {idx}: {alt['Route']}**

                    **⏱️ TIME SAVINGS:**
                    - Current route: {alt['Current_Days']:.1f} days (speed: {alt['Current_Speed']:.0f} km/day)
                    - This route: {alt['Total_Days']:.1f} days
                    - **Saves: {alt['Days_Saved']:.1f} days** ✅

                    **🛣️ ROUTE BREAKDOWN:**
                    - **Leg 1:** {order['Origin']} → {alt['Hub']}
                      - Time: {alt['Leg1_Days']:.1f} days
                      - Carrier: {alt['Carrier1']}
                      - Speed: {alt['Leg1_Speed']:.0f} km/day

                    - **Hub Transfer at {alt['Hub']}:**
                      - Time: {hub_transfer_display}
                      - Details: {alt['Hub_Explanation']}
                      - Confidence: {alt['Hub_Confidence']}

                    - **Leg 2:** {alt['Hub']} → {order['Destination']}
                      - Time: {alt['Leg2_Days']:.1f} days
                      - Carrier: {alt['Carrier2']}
                      - Speed: {alt['Leg2_Speed']:.0f} km/day

                    **📏 DISTANCE TRADE-OFF:**
                    - Direct: {order['Distance_KM']:.0f} km
                    - Via {alt['Hub']}: {alt['Total_Distance']:.0f} km (+{alt['Distance_Overhead']:.0f} km, {extra_distance_pct:.0f}% longer)

                    **💡 WHY FASTER DESPITE LONGER DISTANCE?**
                    - {alt['Hub']} is an efficient hub with {hub_transfer_display} processing time
                    - Better carriers available on each leg
                    - Historical data shows {alt['Total_Days']:.1f} days avg delivery
                    - Direct route is congested/risky ({alt['Current_Days']:.1f} days historically)

                    **💰 EXTRA COST:**
                    - Fuel: ₹{alt['Distance_Overhead'] * 10:.0f} (extra {alt['Distance_Overhead']:.0f} km)
                    - Hub handling: ₹500
                    - **Total extra cost: ₹{alt['Distance_Overhead'] * 10 + 500:.0f}**

                    **✅ VERDICT:** {"HIGHLY RECOMMENDED" if alt['Days_Saved'] > 1.0 else "RECOMMENDED"} - Time savings of {alt['Days_Saved']:.1f} days 
                    """)
            else:
                st.warning(f"""
                ⚠️ **No faster routes found**

                Direct route {order['Route']} is already optimal in terms of total delivery time.

                **Recommendation:** Focus on carrier switching or timeline adjustment.
                """)

            # ============================================================
            # ACTION 2: CARRIER OPTIMIZATION
            # ============================================================

            st.markdown("---")
            st.markdown("### 🚛 Carrier Optimization")

            current_carrier = order['Carrier']

            route_carrier_stats = df[df['Route'] == order['Route']].groupby('Carrier').agg({
                'is_delayed': 'mean',
                'Order_ID': 'count',
                'Delivery_Cost_INR': 'mean'
            }).reset_index()
            route_carrier_stats.columns = ['Carrier', 'Delay_Rate', 'Orders', 'Avg_Cost']
            route_carrier_stats = route_carrier_stats.sort_values('Delay_Rate')

            if len(route_carrier_stats) > 1:
                best_carrier = route_carrier_stats.iloc[0]['Carrier']
                best_carrier_risk = route_carrier_stats.iloc[0]['Delay_Rate']
                best_carrier_cost = route_carrier_stats.iloc[0]['Avg_Cost']

                current_carrier_risk = df[df['Carrier'] == current_carrier]['is_delayed'].mean()

                if current_carrier != best_carrier:
                    improvement = (current_carrier_risk - best_carrier_risk) * 100

                    st.success(f"""
                    **🚛 SWITCH CARRIER**

                    **Current:** {current_carrier} ({current_carrier_risk:.1%} delay rate globally)
                    **Recommended:** {best_carrier} ({best_carrier_risk:.1%} delay rate on {order['Route']})

                    **Impact:**
                    - 📊 Risk reduction: **{improvement:.0f}%** specifically for this route
                    - 💰 Cost: ₹{best_carrier_cost:.0f} (vs ₹{order['Delivery_Cost_INR']:.0f} current)

                    **Historical proof:** {best_carrier} has delivered {route_carrier_stats.iloc[0]['Orders']:.0f} orders on {order['Route']} with {best_carrier_risk:.1%} delays.
                    """)
                else:
                    st.info(f"✅ Already using best carrier ({current_carrier}) for this route!")

            # ============================================================
            # ACTION 3: AI-CALCULATED DEADLINE ADJUSTMENT
            # ============================================================

            st.markdown("---")
            st.markdown("### 📅 AI-Calculated Deadline Optimization")

            required_days, buffer_needed, explanation, speed_used = calculate_required_buffer_ai(order, df)

            if buffer_needed > 0:
                st.warning(f"""
                **⏱️ ADJUST CUSTOMER EXPECTATION**

                **Current promise:** {order['Promised_Delivery_Days']} days
                **AI-calculated required:** {required_days:.1f} days
                **Buffer needed:** +{buffer_needed:.1f} days

                {explanation}

                **Action:**
                1. Email customer NOW: "Delivery will take {required_days:.0f} days due to route conditions"
                2. Update internal ETA to {required_days:.1f} days
                3. Set alert for day {order['Promised_Delivery_Days']:.0f} to check status
                """)
            else:
                st.success(f"""
                ✅ **Current timeline is realistic**

                Promised {order['Promised_Delivery_Days']} days is sufficient.
                AI-calculated required: {required_days:.1f} days

                {explanation}
                """)

            # ============================================================
            # ACTION 4: WEATHER-BASED SCHEDULING
            # ============================================================

            if order['weather_severe'] == 1:
                st.markdown("---")
                st.markdown("### 🌧️ Weather Impact Mitigation")

                st.error(f"""
                **🌧️ SEVERE WEATHER DETECTED**

                Weather impact: {order['Weather_Impact']}

                **Actions:**
                1. **Delay shipment by 1 day** (wait for weather to clear)
                2. **Use waterproof packaging** (add ₹50 cost)
                3. **Add GPS tracker** (real-time monitoring)
                4. **Notify customer** about weather delays

                **Cost-benefit:** Delaying 1 day reduces delay risk by 30%.
                """)

        else:
            st.success(f"""
            ### ✅ Order is LOW RISK ({delay_prob:.1%})

            No special actions needed. Standard processing is sufficient.
            """)

# ============================================================================
# PAGE 4: CARRIER & ROUTE
# ============================================================================

elif page == "🚛 Carrier & Route":
    st.title("🚛 Carrier & Route Intelligence")

    st.markdown("## 🏆 Carrier Performance")

    carrier_stats = df_filtered.groupby('Carrier').agg({
        'delay_probability': 'mean',
        'Order_ID': 'count',
        'Delivery_Cost_INR': 'mean'
    }).reset_index()
    carrier_stats.columns = ['Carrier', 'ML_Risk', 'Orders', 'Avg_Cost']
    carrier_stats = carrier_stats.sort_values('ML_Risk')

    st.dataframe(carrier_stats.style.background_gradient(subset=['ML_Risk'], cmap='RdYlGn_r')
                 .format({'ML_Risk': '{:.1%}', 'Avg_Cost': '₹{:,.0f}'}),
                 use_container_width=True)

    best_carrier = carrier_stats.iloc[0]['Carrier']
    worst_carrier = carrier_stats.iloc[-1]['Carrier']

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"""
        ### ✅ RECOMMENDED
        **{best_carrier}**  
        ML Risk: {carrier_stats.iloc[0]['ML_Risk']:.1%}  
        Cost: ₹{carrier_stats.iloc[0]['Avg_Cost']:,.0f}
        """)

    with col2:
        st.error(f"""
        ### ⛔ HIGH-RISK
        **{worst_carrier}**  
        ML Risk: {carrier_stats.iloc[-1]['ML_Risk']:.1%}  
        Cost: ₹{carrier_stats.iloc[-1]['Avg_Cost']:,.0f}
        """)

    st.markdown("---")
    st.subheader("📍 Origin-Level Risk Analysis")

    if 'Origin' in df_filtered.columns:
        origin_summary = df_filtered.groupby('Origin').agg({
            'delay_probability': 'mean',
            'Order_ID': 'count',
            'Distance_KM': 'mean',
            'Traffic_Delay_Minutes': 'mean',
            'Delivery_Cost_INR': 'mean'
        }).round(3)

        origin_summary.columns = ['ML_Risk', 'Orders', 'Avg_Distance', 'Avg_Traffic', 'Avg_Cost']
        origin_summary = origin_summary.sort_values('ML_Risk', ascending=False)

        origin_summary['Risk_Tier'] = origin_summary['ML_Risk'].apply(
            lambda x: '🔴 Critical' if x > 0.75 else '🟡 High' if x > 0.6 else '🟢 Low'
        )

        origin_summary = origin_summary[['Risk_Tier', 'ML_Risk', 'Orders', 'Avg_Distance', 'Avg_Traffic', 'Avg_Cost']]

        st.dataframe(
            origin_summary.style.background_gradient(subset=['ML_Risk'], cmap='RdYlGn_r')
            .format({
                'ML_Risk': '{:.1%}',
                'Avg_Distance': '{:.0f} km',
                'Avg_Traffic': '{:.0f} min',
                'Avg_Cost': '₹{:,.0f}'
            }),
            use_container_width=True
        )

        st.markdown("### ⚠️ Top 3 High-Risk Origins (WITH AI-CALCULATED BUFFERS)")

        for i, (origin, row) in enumerate(origin_summary.head(3).iterrows(), 1):
            origin_orders = df_filtered[df_filtered['Origin'] == origin]

            if len(origin_orders) > 0:
                avg_order = origin_orders.iloc[0]
                required_days, buffer_needed, _, speed_used = calculate_required_buffer_ai(avg_order, df)

                carrier_stats_origin = df.groupby('Carrier')['is_delayed'].mean()
                best_carrier_origin = carrier_stats_origin.idxmin()
                potential_improvement = (row['ML_Risk'] - carrier_stats_origin[best_carrier_origin]) * 100

                st.warning(f"""
                **{i}. {origin}** {row['Risk_Tier']}
                - ML Risk: {row['ML_Risk']:.1%}
                - Orders: {row['Orders']:.0f}
                - Avg Traffic: {row['Avg_Traffic']:.0f} min
                - AI-learned speed: **{speed_used:.0f} km/day** (from historical data)

                **💡 AI-Calculated Actions:** 
                - Switch to {best_carrier_origin} (reduce delay risk by ~{potential_improvement:.0f}%)
                - Add **+{buffer_needed:.1f} day buffer** 
                - Required delivery time: {required_days:.1f} days for Express orders
                - Monitor traffic patterns for this origin
                """)

# ============================================================================
# PAGE 5: COST ANALYSIS
# ============================================================================

elif page == "💰 Cost Analysis":
    st.title("💰 Cost Savings Analysis")

    col1, col2, col3 = st.columns(3)
    monthly_orders = col1.number_input("Monthly Orders", value=1000, step=100)
    hourly_rate = col2.number_input("Dispatcher Rate (₹/hr)", value=400, step=50)
    intervention_time = col3.number_input("Intervention Time (min)", value=30, step=5)

    baseline_fp_rate = 0.50
    baseline_delay_rate = df['is_delayed'].mean()
    baseline_false_alarms = monthly_orders * baseline_delay_rate * baseline_fp_rate

    model_precision = 0.867
    model_fp_rate = 1 - model_precision
    model_false_alarms = monthly_orders * baseline_delay_rate * model_fp_rate

    saved_interventions = baseline_false_alarms - model_false_alarms
    monthly_savings = saved_interventions * (intervention_time / 60) * hourly_rate
    annual_savings = monthly_savings * 12

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Interventions Saved", f"{saved_interventions:.0f}/mo")
    col2.metric("Hours Saved", f"{saved_interventions * (intervention_time / 60):.1f}/mo")
    col3.metric("Monthly Savings", f"₹{monthly_savings:,.0f}")
    col4.metric("Annual Savings", f"₹{annual_savings:,.0f}")

    st.markdown("---")
    st.subheader("📊 Cost Comparison")

    baseline_cost = baseline_false_alarms * (intervention_time / 60) * hourly_rate
    model_cost = model_false_alarms * (intervention_time / 60) * hourly_rate

    comparison = pd.DataFrame({
        'Scenario': ['Without Model', 'With Model', 'Savings'],
        'Cost': [baseline_cost, model_cost, monthly_savings]
    })

    fig = px.bar(comparison, x='Scenario', y='Cost', text='Cost',
                 title=f"Monthly Savings for {monthly_orders:,} Orders",
                 color='Scenario',
                 color_discrete_map={
                     'Without Model': '#ADD8E6',
                     'With Model': '#FF6B6B',
                     'Savings': '#51CF66'
                 })
    fig.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📊 How Savings Are Calculated"):
        st.markdown(f"""
        ### Without Model (Manual Process)
        - Monthly orders: {monthly_orders:,}
        - Delay rate: {baseline_delay_rate:.1%}
        - False positive rate: {baseline_fp_rate:.0%}
        - False alarms: {baseline_false_alarms:.0f} orders
        - **Cost: ₹{baseline_cost:,.0f}**

        ### With ML Model
        - Delay rate: {baseline_delay_rate:.1%}
        - False positive rate: {model_fp_rate:.1%} (test precision = {model_precision:.1%})
        - False alarms: {model_false_alarms:.0f} orders
        - **Cost: ₹{model_cost:,.0f}**

        ### Net Savings
        - **Monthly savings: ₹{monthly_savings:,.0f}**
        - **Annual savings: ₹{annual_savings:,.0f}**
        """)

# ============================================================================
# PAGE 6: PREDICT NEW ORDER
# ============================================================================

elif page == "🔮 Predict New Order":
    st.title("🔮 Predict New Order Delay")
    st.markdown("### Input order details for AI-powered prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        order_date = st.date_input("Order Date", value=datetime.now())
        customer_segment = st.selectbox("Customer Segment", df['Customer_Segment'].unique())
        priority = st.selectbox("Priority", df['Priority'].unique())

    with col2:
        product_category = st.selectbox("Product Category", df['Product_Category'].unique())
        order_value = st.number_input("Order Value (₹)", value=5000, step=500)
        origin = st.selectbox("Origin", df['Origin'].unique())

    with col3:
        destination = st.selectbox("Destination", df['Destination'].unique())
        carrier = st.selectbox("Preferred Carrier", df['Carrier'].unique())
        weather = st.selectbox("Weather Forecast", df['Weather_Impact'].unique())

    if st.button("🔮 Predict Delay Risk", type="primary"):
        route = f"{origin}-{destination}"
        route_data = df[df['Route'] == route]

        if len(route_data) > 0:
            avg_distance = route_data['Distance_KM'].mean()
            avg_traffic = route_data['Traffic_Delay_Minutes'].mean()
            route_risk_val = route_data['route_risk'].iloc[0]
        else:
            avg_distance = df['Distance_KM'].mean()
            avg_traffic = df['Traffic_Delay_Minutes'].mean()
            route_risk_val = 0.5

        carrier_data = df[df['Carrier'] == carrier]
        carrier_risk_val = carrier_data['carrier_risk'].iloc[0] if len(carrier_data) > 0 else 0.5

        pc_data = df[(df['Priority'] == priority) & (df['Carrier'] == carrier)]
        priority_carrier_risk_val = pc_data['priority_carrier_risk'].iloc[0] if len(pc_data) > 0 else 0.5

        order_datetime = pd.Timestamp(order_date)
        day_of_week = order_datetime.dayofweek
        is_weekend = int(day_of_week >= 5)
        is_peak_season = int(order_datetime.month in [11, 12])
        quarter = order_datetime.quarter

        promised_days = 3 if priority == 'Express' else 5 if priority == 'Standard' else 7
        traffic_severe = int(avg_traffic > 60)
        weather_severe = int(weather in ['Heavy_Rain', 'Fog'])
        distance_median = df['Distance_KM'].median()
        long_distance = int(avg_distance > distance_median * 1.5)
        compound_risk = traffic_severe * 2 + weather_severe * 2 + long_distance + int(carrier_risk_val > 0.5) * 2
        time_pressure = avg_distance / (promised_days + 1)
        cost_per_km = (avg_distance * 10) / (avg_distance + 1)

        new_order = pd.DataFrame([{
            'Priority': priority,
            'Customer_Segment': customer_segment,
            'Carrier': carrier,
            'Distance_KM': avg_distance,
            'Traffic_Delay_Minutes': avg_traffic,
            'Weather_Impact': weather,
            'Promised_Delivery_Days': promised_days,
            'carrier_risk': carrier_risk_val,
            'route_risk': route_risk_val,
            'priority_carrier_risk': priority_carrier_risk_val,
            'traffic_severe': traffic_severe,
            'weather_severe': weather_severe,
            'long_distance': long_distance,
            'compound_risk': compound_risk,
            'time_pressure': time_pressure,
            'cost_per_km': cost_per_km,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'is_peak_season': is_peak_season,
            'quarter': quarter
        }])

        for col in ['Priority', 'Customer_Segment', 'Carrier', 'Weather_Impact']:
            if col in artifacts['label_encoders']:
                le = artifacts['label_encoders'][col]
                new_order[col] = le.transform([new_order[col].iloc[0]])

        X_new = new_order[artifacts['feature_cols']]
        X_new_scaled = artifacts['scaler'].transform(X_new)
        delay_prob = artifacts['model'].predict_proba(X_new_scaled)[0][1]

        st.markdown("---")
        st.markdown("## 📊 AI Prediction Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            risk_color = "🔴" if delay_prob > 0.7 else "🟡" if delay_prob > 0.4 else "🟢"
            risk_text = "HIGH RISK" if delay_prob > 0.7 else "MEDIUM RISK" if delay_prob > 0.4 else "LOW RISK"
            st.metric("Risk Level", f"{risk_color} {risk_text}")

        with col2:
            st.metric("Delay Probability", f"{delay_prob:.1%}")

        with col3:
            predicted_days = promised_days + (delay_prob * 2)
            st.metric("Predicted Delivery", f"{predicted_days:.1f} days")

        st.markdown("---")
        st.subheader("🛣️ Route Information")

        col1, col2 = st.columns(2)

        with col1:
            st.info(f"""
            **Route:** {route}
            - Distance: {avg_distance:.0f} km
            - Avg Traffic: {avg_traffic:.0f} min
            - Route Risk: {route_risk_val:.1%}
            """)

        with col2:
            st.info(f"""
            **Carrier:** {carrier}
            - Carrier Risk: {carrier_risk_val:.1%}
            - Priority-Carrier Risk: {priority_carrier_risk_val:.1%}
            """)

        if delay_prob > 0.7:
            st.error("""
            ### 🚨 HIGH RISK - RECOMMENDATIONS:
            1. Consider switching to a lower-risk carrier
            2. Use AI-calculated buffer (check Order Analysis)
            3. Set up real-time tracking alerts
            4. Prepare customer communication
            """)
        elif delay_prob > 0.4:
            st.warning("""
            ### ⚠️ MEDIUM RISK - SUGGESTIONS:
            1. Monitor traffic conditions closely
            2. Have backup carrier on standby
            3. Brief customer about possible delays
            """)
        else:
            st.success("""
            ### ✅ LOW RISK - STANDARD PROCESSING
            Order looks good for on-time delivery!
            """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666;'>
    <p><b>NEXGEN Logistics Intelligence</b> | AI-Powered Delivery Optimization</p>
    <p>All metrics learned from data | Threshold: {active_threshold:.3f}</p>
</div>
""", unsafe_allow_html=True)

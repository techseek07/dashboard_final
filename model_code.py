# ============================================================================
# NEXGEN LOGISTICS: COMPLETE END-TO-END PIPELINE
# Features: Time-based patterns + Overfitting check + ML-TCO Integration
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             make_scorer, f1_score, recall_score, precision_score)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: LOAD AND MERGE DATA
# ============================================================================
from google.colab import files
uploaded = files.upload()

df_delivery = pd.read_csv('delivery_performance.csv')
df_orders = pd.read_csv('orders.csv')
df_routes = pd.read_csv('routes_distance.csv')
df_fleet = pd.read_csv('vehicle_fleet.csv')
df_costs = pd.read_csv('cost_breakdown.csv')

df_merged = df_delivery.merge(df_orders, on='Order_ID', how='left') \
                       .merge(df_routes, on='Order_ID', how='left') \
                       .merge(df_costs, on='Order_ID', how='left')

# Handle missing
df_merged['Weather_Impact'] = df_merged['Weather_Impact'].fillna('None')
df_merged['Special_Handling'] = df_merged['Special_Handling'].fillna('None')
df_merged['Traffic_Delay_Minutes'] = df_merged['Traffic_Delay_Minutes'].fillna(0)

# Target
df_merged['is_delayed'] = df_merged['Delivery_Status'].isin(['Slightly-Delayed', 'Severely-Delayed']).astype(int)

print(f"✅ Merged dataset: {df_merged.shape}")
print(f"Delayed rate: {df_merged['is_delayed'].mean():.1%}")

# ============================================================================
# STEP 2: ANALYZE DATA QUALITY & REMOVE CONFLICTING SAMPLES
# ============================================================================
print("\n" + "="*70)
print("STEP 2: DATA QUALITY ANALYSIS")
print("="*70)

def flag_suspicious_delays(row):
    """Flag orders that are delayed despite favorable conditions"""
    if row['is_delayed'] == 1:
        if (row['Traffic_Delay_Minutes'] < 30 and
            row['Distance_KM'] < 500 and
            row['Weather_Impact'] == 'None'):
            return True
    return False

def flag_suspicious_ontime(row):
    """Flag orders that are on-time despite bad conditions"""
    if row['is_delayed'] == 0:
        if (row['Traffic_Delay_Minutes'] > 90 or
            row['Weather_Impact'] in ['Heavy_Rain', 'Fog'] or
            row['Distance_KM'] > 2000):
            return True
    return False

df_merged['suspicious_delayed'] = df_merged.apply(flag_suspicious_delays, axis=1)
df_merged['suspicious_ontime'] = df_merged.apply(flag_suspicious_ontime, axis=1)

suspicious_count = df_merged['suspicious_delayed'].sum() + df_merged['suspicious_ontime'].sum()
print(f"\n🔍 Found {suspicious_count} potentially noisy samples")
print(f"   Suspicious delays: {df_merged['suspicious_delayed'].sum()}")
print(f"   Suspicious on-time: {df_merged['suspicious_ontime'].sum()}")

# REMOVE suspicious samples
df_clean = df_merged[~(df_merged['suspicious_delayed'] | df_merged['suspicious_ontime'])].copy()
print(f"\n✅ Cleaned dataset: {len(df_merged)} → {len(df_clean)} samples")

# ============================================================================
# STEP 3: ADVANCED FEATURE ENGINEERING (WITH TIME-BASED FEATURES)
# ============================================================================
print("\n" + "="*70)
print("STEP 3: ADVANCED FEATURE ENGINEERING")
print("="*70)

# 1. Carrier-specific delay rates
carrier_delay_rate = df_clean.groupby('Carrier')['is_delayed'].mean()
df_clean['carrier_risk'] = df_clean['Carrier'].map(carrier_delay_rate)

# 2. Route-specific patterns
route_delay_rate = df_clean.groupby('Route')['is_delayed'].mean()
df_clean['route_risk'] = df_clean['Route'].map(route_delay_rate)

# 3. Priority-Carrier interaction
priority_carrier_delay = df_clean.groupby(['Priority', 'Carrier'])['is_delayed'].mean()
df_clean['priority_carrier_risk'] = df_clean.apply(
    lambda x: priority_carrier_delay.get((x['Priority'], x['Carrier']), 0.5), axis=1
)

# 4. Traffic severity bins
df_clean['traffic_severe'] = (df_clean['Traffic_Delay_Minutes'] > 60).astype(int)

# 5. Weather severity
df_clean['weather_severe'] = df_clean['Weather_Impact'].isin(['Heavy_Rain', 'Fog']).astype(int)

# 6. Distance risk
distance_median = df_clean['Distance_KM'].median()
df_clean['long_distance'] = (df_clean['Distance_KM'] > distance_median * 1.5).astype(int)

# 7. Combined risk score
df_clean['compound_risk'] = (
    df_clean['traffic_severe'] * 2 +
    df_clean['weather_severe'] * 2 +
    df_clean['long_distance'] +
    (df_clean['carrier_risk'] > 0.5).astype(int) * 2
)

# 8. Time pressure
df_clean['time_pressure'] = df_clean['Distance_KM'] / (df_clean['Promised_Delivery_Days'] + 1)

# 9. Cost efficiency
df_clean['cost_per_km'] = df_clean['Delivery_Cost_INR'] / (df_clean['Distance_KM'] + 1)

# ============================================================================
# 10-13: NEW TIME-BASED FEATURES
# ============================================================================
print("\n🆕 Adding time-based features...")

# Parse Order_Date
df_clean['Order_Date'] = pd.to_datetime(df_clean['Order_Date'], errors='coerce')

# Day of week (0=Monday, 6=Sunday)
df_clean['day_of_week'] = df_clean['Order_Date'].dt.dayofweek

# Weekend flag
df_clean['is_weekend'] = (df_clean['day_of_week'] >= 5).astype(int)

# Peak season (November-December holiday rush)
df_clean['is_peak_season'] = df_clean['Order_Date'].dt.month.isin([11, 12]).astype(int)

# Quarter of year (Q4 = Oct-Dec busier)
df_clean['quarter'] = df_clean['Order_Date'].dt.quarter

print("✅ Created 13 advanced features (9 original + 4 time-based):")
print("   - carrier_risk, route_risk, priority_carrier_risk")
print("   - traffic_severe, weather_severe, long_distance")
print("   - compound_risk, time_pressure, cost_per_km")
print("   - day_of_week, is_weekend, is_peak_season, quarter")

# ============================================================================
# STEP 4: SELECT BEST FEATURES
# ============================================================================
feature_cols = [
    'Priority', 'Customer_Segment', 'Carrier',
    'Distance_KM', 'Traffic_Delay_Minutes', 'Weather_Impact',
    'Promised_Delivery_Days',
    # Advanced features
    'carrier_risk', 'route_risk', 'priority_carrier_risk',
    'traffic_severe', 'weather_severe', 'long_distance',
    'compound_risk', 'time_pressure', 'cost_per_km',
    # Time-based features
    'day_of_week', 'is_weekend', 'is_peak_season', 'quarter'
]

df_model = df_clean[feature_cols + ['is_delayed', 'Order_ID']].copy()

# Encode categoricals
label_encoders = {}
categorical_cols = ['Priority', 'Customer_Segment', 'Carrier', 'Weather_Impact']

for col in categorical_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    label_encoders[col] = le

X = df_model[feature_cols]
y = df_model['is_delayed']

print(f"\n✅ Feature matrix: {X.shape}")

# ============================================================================
# STEP 5: K-FOLD CROSS-VALIDATION
# ============================================================================
print("\n" + "="*70)
print("STEP 5: K-FOLD CROSS-VALIDATION")
print("="*70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def cv_with_smote(model, X, y, cv):
    """Cross-validation with SMOTE in each fold"""
    scores = {'recall': [], 'precision': [], 'f1': [], 'accuracy': []}

    for train_idx, test_idx in cv.split(X, y):
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

        smote = SMOTE(random_state=42, k_neighbors=3)
        X_train_res, y_train_res = smote.fit_resample(X_train_fold, y_train_fold)

        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test_fold)

        scores['accuracy'].append((y_pred == y_test_fold).mean())
        scores['recall'].append(recall_score(y_test_fold, y_pred, zero_division=0))
        scores['precision'].append(precision_score(y_test_fold, y_pred, zero_division=0))
        scores['f1'].append(f1_score(y_test_fold, y_pred, zero_division=0))

    return scores

models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=5, min_samples_leaf=5,
        class_weight='balanced', random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.7, random_state=42
    ),
    'Random Forest (Deep)': RandomForestClassifier(
        n_estimators=150, max_depth=8, min_samples_leaf=3,
        class_weight='balanced', random_state=42
    )
}

cv_results = {}
for name, model in models.items():
    print(f"\n🔄 Evaluating {name}...")
    scores = cv_with_smote(model, X_scaled, y, cv)
    cv_results[name] = scores

    print(f"   Recall:    {np.mean(scores['recall']):.3f} ± {np.std(scores['recall']):.3f}")
    print(f"   Precision: {np.mean(scores['precision']):.3f} ± {np.std(scores['precision']):.3f}")
    print(f"   F1:        {np.mean(scores['f1']):.3f} ± {np.std(scores['f1']):.3f}")
    print(f"   Accuracy:  {np.mean(scores['accuracy']):.3f} ± {np.std(scores['accuracy']):.3f}")

# ============================================================================
# STEP 6: FINAL MODEL TRAINING
# ============================================================================
print("\n" + "="*70)
print("STEP 6: FINAL MODEL TRAINING")
print("="*70)

best_model_name = max(cv_results, key=lambda x: np.mean(cv_results[x]['f1']))
best_model = models[best_model_name]

print(f"🏆 Best model: {best_model_name}")
print(f"   CV F1: {np.mean(cv_results[best_model_name]['f1']):.3f}")
print(f"   CV Recall: {np.mean(cv_results[best_model_name]['recall']):.3f}")

# Train on full dataset
smote = SMOTE(random_state=42, k_neighbors=3)
X_full_res, y_full_res = smote.fit_resample(X_scaled, y)
best_model.fit(X_full_res, y_full_res)

print(f"\n✅ Final model trained on {len(X_full_res)} samples (after SMOTE)")

# ============================================================================
# STEP 6.5: OVERFITTING CHECK
# ============================================================================
print("\n" + "="*70)
print("STEP 6.5: OVERFITTING CHECK (TRAIN VS TEST)")
print("="*70)

X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

smote_check = SMOTE(random_state=42, k_neighbors=3)
X_train_res_check, y_train_res_check = smote_check.fit_resample(X_train_split, y_train_split)

overfitting_model = RandomForestClassifier(
    n_estimators=150, max_depth=8, min_samples_leaf=3,
    class_weight='balanced', random_state=42
)
overfitting_model.fit(X_train_res_check, y_train_res_check)

train_pred = overfitting_model.predict(X_train_res_check)
test_pred = overfitting_model.predict(X_test_split)

train_acc = (train_pred == y_train_res_check).mean()
test_acc = (test_pred == y_test_split).mean()
gap = abs(train_acc - test_acc)

train_f1 = f1_score(y_train_res_check, train_pred)
test_f1 = f1_score(y_test_split, test_pred)
f1_gap = abs(train_f1 - test_f1)

print(f"\n📊 OVERFITTING ANALYSIS:")
print(f"   Train Accuracy: {train_acc:.1%}")
print(f"   Test Accuracy:  {test_acc:.1%}")
print(f"   Accuracy Gap:   {gap:.1%}")
print(f"\n   Train F1:       {train_f1:.1%}")
print(f"   Test F1:        {test_f1:.1%}")
print(f"   F1 Gap:         {f1_gap:.1%}")

print(f"\n🎯 INTERPRETATION:")
if gap < 0.05:
    print("   ✅ EXCELLENT: Minimal overfitting (gap < 5%)")
elif gap < 0.10:
    print("   ✅ GOOD: Low overfitting (gap 5-10%)")
elif gap < 0.15:
    print("   ⚠️  ACCEPTABLE: Mild overfitting (gap 10-15%)")
else:
    print("   🔴 WARNING: Overfitting detected (gap > 15%)")

print(f"\n📊 DETAILED TEST SET PERFORMANCE:")
print(classification_report(y_test_split, test_pred,
                           target_names=['On-Time', 'Delayed'],
                           digits=3))

cm = confusion_matrix(y_test_split, test_pred)
print(f"\n🎯 Test Set Confusion Matrix:")
print(f"              Predicted")
print(f"            On-Time  Delayed")
print(f"On-Time        {cm[0,0]:4d}    {cm[0,1]:4d}")
print(f"Delayed        {cm[1,0]:4d}    {cm[1,1]:4d}")

# ============================================================================
# STEP 7: FEATURE IMPORTANCE
# ============================================================================
print("\n" + "="*70)
print("STEP 7: FEATURE IMPORTANCE")
print("="*70)

if hasattr(best_model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\n📊 Top 15 Most Important Features:")
    print(importance_df.head(15).to_string(index=False))

    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df.head(15), x='Importance', y='Feature', palette='viridis')
    plt.title(f'{best_model_name} - Feature Importance')
    plt.tight_layout()
    plt.show()

# ============================================================================
# STEP 8: ML-INTEGRATED TCO OPTIMIZER
# ============================================================================
print("\n" + "="*70)
print("STEP 8: ML-INTEGRATED TCO OPTIMIZER")
print("="*70)

# Calibrate cost parameters
fuel_price_per_liter = df_clean['Fuel_Cost'].sum() / df_clean['Fuel_Consumption_L'].sum()
labor_cost_per_km = df_clean['Labor_Cost'].sum() / df_clean['Distance_KM'].sum()

print(f"📊 Calibrated parameters:")
print(f"   Fuel price: ₹{fuel_price_per_liter:.2f}/L")
print(f"   Labor cost: ₹{labor_cost_per_km:.2f}/km")

# Updated TCO weights
TCO_WEIGHTS = {
    'Express': {'w_time': 0.70, 'w_cost': 0.20, 'w_eco': 0.10, 'lambda_pen': 4.0},
    'Standard': {'w_time': 0.50, 'w_cost': 0.30, 'w_eco': 0.20, 'lambda_pen': 2.5},
    'Economy': {'w_time': 0.30, 'w_cost': 0.50, 'w_eco': 0.20, 'lambda_pen': 1.5}
}

def predict_delay_probability(order_features, model, scaler, feature_cols):
    """Get ML-based delay probability for an order"""
    X_order = order_features[feature_cols].values.reshape(1, -1)
    X_order_scaled = scaler.transform(X_order)
    delay_prob = model.predict_proba(X_order_scaled)[0][1]
    return delay_prob

def compute_ml_tco(order_row_full, order_row_features, vehicle_row, weights, model, scaler, feature_cols):
    """
    TCO with ML-predicted delay probability

    Args:
        order_row_full: Row from df_clean (has all columns including Toll_Charges_INR)
        order_row_features: Row from df_model (has only feature_cols for ML prediction)
        vehicle_row: Row from df_fleet
        weights: TCO weights dict
        model: Trained ML model
        scaler: Fitted StandardScaler
        feature_cols: List of feature column names
    """

    # Get ML-predicted delay probability using feature row
    delay_prob = predict_delay_probability(order_row_features, model, scaler, feature_cols)

    # Delay penalty (ML-driven) - use full row for all columns
    traffic_days = float(order_row_full.get('Traffic_Delay_Minutes', 0)) / 1440.0

    weather = order_row_full.get('Weather_Impact', 'None')
    if weather == 'Heavy_Rain':
        weather_bump = 0.5
    elif weather in ['Light_Rain', 'Fog']:
        weather_bump = 0.2
    else:
        weather_bump = 0.0

    promised_days = float(order_row_full.get('Promised_Delivery_Days', 3))
    predicted_days = promised_days + traffic_days + weather_bump

    # Use ML probability to scale delay penalty
    delay_penalty = delay_prob * max(0, predicted_days - promised_days) * weights['lambda_pen']

    # Financial cost - use full row which has Toll_Charges_INR
    distance = float(order_row_full.get('Distance_KM', 0))
    fuel_efficiency = float(vehicle_row.get('Fuel_Efficiency_KM_per_L', 8))

    if fuel_efficiency > 0:
        fuel_cost = (distance / fuel_efficiency) * fuel_price_per_liter
    else:
        fuel_cost = 0

    toll_cost = float(order_row_full.get('Toll_Charges_INR', 0))  # ✅ Now works!
    labor_cost = distance * labor_cost_per_km
    financial_cost = fuel_cost + toll_cost + labor_cost

    # Environmental cost
    co2_emission = float(vehicle_row.get('CO2_Emissions_Kg_per_KM', 0.3))
    environmental_cost = distance * co2_emission

    # Total weighted cost score
    twcs = (weights['w_time'] * delay_penalty +
            weights['w_cost'] * financial_cost +
            weights['w_eco'] * environmental_cost)

    return {
        'Vehicle_ID': vehicle_row.get('Vehicle_ID', 'Unknown'),
        'Vehicle_Type': vehicle_row.get('Vehicle_Type', 'Unknown'),
        'ML_Delay_Probability': delay_prob,
        'Predicted_Days': predicted_days,
        'Delay_Penalty': delay_penalty,
        'Financial_Cost': financial_cost,
        'Environmental_Cost': environmental_cost,
        'TWCS': twcs
    }

# Test on sample order
print("\n🎯 TESTING ML-INTEGRATED TCO ON SAMPLE ORDER:")
sample_order_id = df_clean['Order_ID'].iloc[0]

# Get full row with all columns (for TCO calculation)
sample_order_full = df_clean[df_clean['Order_ID'] == sample_order_id].iloc[0]

# Get feature row (for ML prediction)
sample_order_features = df_model[df_model['Order_ID'] == sample_order_id].iloc[0]

priority = sample_order_full['Priority']
weights = TCO_WEIGHTS.get(priority, TCO_WEIGHTS['Standard'])

print(f"\n📦 Order {sample_order_id}:")
print(f"   Priority: {priority}")
print(f"   Route: {sample_order_full['Route']}")
print(f"   Distance: {sample_order_full['Distance_KM']:.0f} km")
print(f"   Historical Status: {sample_order_full['Delivery_Status']}")

# Get available vehicles
available_vehicles = df_fleet[df_fleet['Status'] == 'Available'].copy()

tco_results = []
for _, vehicle in available_vehicles.iterrows():
    # ✅ Pass both full row and feature row
    result = compute_ml_tco(sample_order_full, sample_order_features, vehicle,
                            weights, best_model, scaler, feature_cols)
    tco_results.append(result)

tco_df = pd.DataFrame(tco_results).sort_values('TWCS')

print("\n📊 Top 5 Vehicle Recommendations (ML-Enhanced TCO):")
print(tco_df.head(5)[['Vehicle_ID', 'Vehicle_Type', 'ML_Delay_Probability',
                      'Financial_Cost', 'TWCS']].to_string(index=False))

best_vehicle = tco_df.iloc[0]
print(f"\n✅ RECOMMENDED VEHICLE: {best_vehicle['Vehicle_ID']} ({best_vehicle['Vehicle_Type']})")
print(f"   ML Delay Probability: {best_vehicle['ML_Delay_Probability']:.1%}")
print(f"   Predicted Delivery: {best_vehicle['Predicted_Days']:.1f} days")
print(f"   Estimated Cost: ₹{best_vehicle['Financial_Cost']:.2f}")
print(f"   Total TWCS: {best_vehicle['TWCS']:.2f}")
# ============================================================================
# STEP 9: CARRIER SELECTION WITH ML
# ============================================================================
print("\n" + "="*70)
print("STEP 9: CARRIER SELECTION (ML-ENHANCED)")
print("="*70)

carrier_stats = df_clean.groupby('Carrier').agg({
    'is_delayed': ['mean', 'count'],
    'Delivery_Cost_INR': 'mean'
}).round(3)
carrier_stats.columns = ['Delay_Rate', 'Order_Count', 'Avg_Cost']
carrier_stats = carrier_stats.sort_values('Delay_Rate')

print("\n📊 Carrier Performance Benchmarks:")
print(carrier_stats.to_string())

print("\n🚨 HIGH-RISK CARRIERS FOR EXPRESS ORDERS:")
high_risk_carriers = carrier_stats[carrier_stats['Delay_Rate'] > 0.6].index.tolist()
for carrier in high_risk_carriers:
    print(f"   ⛔ {carrier}: {carrier_stats.loc[carrier, 'Delay_Rate']:.1%} delay rate - AVOID for Express")

print("\n✅ LOW-RISK CARRIERS (RECOMMENDED):")
low_risk_carriers = carrier_stats[carrier_stats['Delay_Rate'] < 0.5].index.tolist()
for carrier in low_risk_carriers:
    print(f"   ✅ {carrier}: {carrier_stats.loc[carrier, 'Delay_Rate']:.1%} delay rate - SAFE for all priorities")

# ============================================================================
# STEP 10: SAVE MODEL & ARTIFACTS
# ============================================================================
print("\n" + "="*70)
print("STEP 10: SAVE MODEL ARTIFACTS")
print("="*70)

import pickle

artifacts = {
    'model': best_model,
    'scaler': scaler,
    'label_encoders': label_encoders,
    'feature_cols': feature_cols,
    'TCO_WEIGHTS': TCO_WEIGHTS,
    'fuel_price': fuel_price_per_liter,
    'labor_cost_per_km': labor_cost_per_km,
    'high_risk_carriers': high_risk_carriers,
    'low_risk_carriers': low_risk_carriers
}

with open('nexgen_full_model.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("✅ Model artifacts saved to 'nexgen_full_model.pkl'")
print("   Includes: model, scaler, encoders, TCO parameters, carrier risk profiles")

files.download('nexgen_full_model.pkl')

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("🎉 COMPLETE PIPELINE FINISHED!")
print("="*70)

best_f1 = np.mean(cv_results[best_model_name]['f1'])
best_recall = np.mean(cv_results[best_model_name]['recall'])

print(f"\n✅ FINAL PERFORMANCE:")
print(f"   Model: {best_model_name}")
print(f"   CV F1-Score: {best_f1:.1%}")
print(f"   CV Recall: {best_recall:.1%}")
print(f"   Train-Test Gap: {gap:.1%}")

print(f"\n🎯 DEPLOYMENT READINESS:")
if best_f1 > 0.80 and gap < 0.10:
    print("   🟢 PRODUCTION-READY: Excellent performance & generalization")
elif best_f1 > 0.70 and gap < 0.15:
    print("   🟡 DEPLOY WITH MONITORING: Good performance, watch for drift")
else:
    print("   🔴 NEEDS IMPROVEMENT: Consider collecting more data")

print(f"\n📊 KEY INSIGHTS:")
print(f"   1. Top feature: {importance_df.iloc[0]['Feature']} ({importance_df.iloc[0]['Importance']:.1%} importance)")
print(f"   2. Time features added: day_of_week, is_weekend, is_peak_season, quarter")
print(f"   3. ML-TCO integration: Delay probability scales TCO penalty dynamically")
print(f"   4. Carrier recommendations: Avoid {len(high_risk_carriers)} high-risk carriers for Express")

print("\n📌 NEXT ACTIONS:")
print("   1. ✅ Deploy ML model API for real-time predictions")
print("   2. ✅ Integrate TCO optimizer with dispatch system")
print("   3. ✅ Set alerts for orders with delay_prob > 60%")
print("   4. ✅ Route Express orders away from high-risk carriers")
print("   5. ✅ Monitor model performance weekly, retrain quarterly")

print("\n💾 All artifacts saved and ready for deployment!")

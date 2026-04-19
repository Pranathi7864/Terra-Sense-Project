import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import pickle
import time
import json
from datetime import datetime

st.set_page_config(
    page_title="TERRA-SENSE",
    layout="wide",
    page_icon="🌍"
)


st.markdown("""
<style>
.main { background-color: #0a0a0a; }
.stMetric { background-color: #1a1a2e; padding: 10px; border-radius: 8px; }
.alert-critical { background-color: rgba(255,0,0,0.2); border-left: 4px solid red; padding: 10px; }
.alert-warning  { background-color: rgba(255,136,0,0.2); border-left: 4px solid orange; padding: 10px; }
.alert-stable   { background-color: rgba(0,255,0,0.2); border-left: 4px solid green; padding: 10px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    xgb = pickle.load(open('ml/car_model.pkl', 'rb'))
    iso = pickle.load(open('ml/isolation_model.pkl', 'rb'))
    return xgb, iso

xgb_model, iso_model = load_models()


@st.cache_data
def load_data():
    df = pd.read_csv('data/singrauli_sensor_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

df = load_data()

legal_mines = {
    "NTPC_Vindhyachal": {"lat": 24.1100, "lon": 82.6600, "radius_km": 2.0, "operator": "NTPC Ltd"},
    "NCL_Jhingurda":    {"lat": 24.0900, "lon": 82.6400, "radius_km": 1.5, "operator": "NCL"},
    "NCL_Amlohri":      {"lat": 24.1450, "lon": 82.6550, "radius_km": 1.8, "operator": "NCL"},
    "Reliance_Moher":   {"lat": 24.1300, "lon": 82.5900, "radius_km": 1.2, "operator": "Reliance"},
}

def get_risk(car):
    if car >= 0.65:   return "STABLE",   "success", "🟢"
    elif car >= 0.40: return "WATCH",    "warning", "🟡"
    elif car >= 0.25: return "WARNING",  "warning", "🟠"
    else:             return "CRITICAL", "error",   "🔴"

def get_recommendation(car, zone):
    if car >= 0.65:
        return "✅ No action needed. Continue monitoring."
    elif car >= 0.40:
        return f"⚠️ {zone}: Monitor closely. Begin afforestation planning."
    elif car >= 0.25:
        return (f"🟠 {zone}: Plant Vetiver grass immediately. "
                f"Improve drainage. Schedule backfilling.")
    else:
        return (f"🚨 {zone}: IMMEDIATE — Reroute defense convoys. "
                f"Deploy geo-textile. Emergency Vetiver planting "
                f"(Chrysopogon zizanioides) across 2.3km². "
                f"Backfill subsurface cavities.")

features = [
    'soil_moisture', 'co2_ppm', 'temperature',
    'humidity', 'vibration', 'ndvi',
    'rainfall_mm', 'mining_activity'
]

col_logo, col_title = st.columns([1, 8])
with col_title:
    st.markdown("# 🌍 TERRA-SENSE")
    st.markdown("**CAR Monitoring System | Singrauli Defense Corridor | CSIR-CMERI Track 2**")

st.divider()


st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Flag_of_India.svg/320px-Flag_of_India.svg.png", width=100)
st.sidebar.markdown("## ⚙️ Controls")

mode = st.sidebar.radio(
    "Demo Mode",
    ["📊 Simulation Mode",
     "📡 Live Sensor Mode",
     "🕵️ Illegal Mining Demo"]
)

zone = st.sidebar.selectbox(
    "Select Zone",
    df['zone'].unique()
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Zone CAR Summary")
zone_car = df.groupby('zone')['CAR'].mean()
for z, c in zone_car.items():
    risk, _, icon = get_risk(c)
    st.sidebar.markdown(f"{icon} **{z}**: {c:.3f} — {risk}")


zone_df  = df[df['zone'] == zone].copy()
latest   = zone_df.iloc[-1]
prev     = zone_df.iloc[-2]
car_now  = latest['CAR']
car_prev = prev['CAR']
risk_label, risk_type, risk_icon = get_risk(car_now)


st.markdown(f"### {risk_icon} Zone: {zone} — Current Status")
m1, m2, m3, m4, m5 = st.columns(5)

m1.metric("CAR Value",     f"{car_now:.3f}",
          delta=f"{car_now - car_prev:.3f}")
m2.metric("CO₂ (ppm)",     f"{latest['co2_ppm']:.0f}")
m3.metric("Soil Moisture", f"{latest['soil_moisture']:.1f}%")
m4.metric("Temperature",   f"{latest['temperature']:.1f}°C")
m5.metric("Vibration",     "YES" if latest['vibration'] else "NO")

st.divider()

if risk_type == "error":
    st.error(f"🚨 CRITICAL ALERT — {zone} | CAR = {car_now:.3f} | Terrain Collapse Risk HIGH")
elif risk_type == "warning":
    st.warning(f"⚠️ {risk_label} ALERT — {zone} | CAR = {car_now:.3f}")
else:
    st.success(f"✅ {zone} — STABLE | CAR = {car_now:.3f}")

st.info(f"📋 **Recommendation:** {get_recommendation(car_now, zone)}")
st.divider()

left, right = st.columns(2)

with left:
    st.markdown("### 📈 CAR Trend — 6 Month History")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=zone_df['timestamp'],
        y=zone_df['CAR'],
        mode='lines',
        name='CAR Value',
        line=dict(color='#00ff88', width=2),
        fill='tozeroy',
        fillcolor='rgba(0,255,136,0.1)'
    ))
    fig.add_hline(y=0.25, line_color="red",
                  line_dash="dash",
                  annotation_text="🔴 Critical (0.25)")
    fig.add_hline(y=0.40, line_color="orange",
                  line_dash="dash",
                  annotation_text="🟠 Warning (0.40)")
    fig.add_hline(y=0.65, line_color="green",
                  line_dash="dash",
                  annotation_text="🟢 Stable (0.65)")
    fig.update_layout(
        height=380,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='#333333'),
        yaxis=dict(gridcolor='#333333', range=[0, 1]),
        xaxis_title="Time",
        yaxis_title="CAR Value"
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("### 📊 All Zones — Risk Comparison")
    zone_summary = df.groupby('zone').agg(
        CAR=('CAR', 'mean'),
        CO2=('co2_ppm', 'mean'),
        Moisture=('soil_moisture', 'mean')
    ).reset_index()

    colors = [
        '#ff0000' if c < 0.25
        else '#ff8800' if c < 0.40
        else '#ffff00' if c < 0.65
        else '#00ff00'
        for c in zone_summary['CAR']
    ]

    fig2 = go.Figure(go.Bar(
        x=zone_summary['zone'],
        y=zone_summary['CAR'],
        marker_color=colors,
        text=[f"{c:.2f}" for c in zone_summary['CAR']],
        textposition='outside'
    ))
    fig2.add_hline(y=0.25, line_color="red",   line_dash="dash")
    fig2.add_hline(y=0.65, line_color="green", line_dash="dash")
    fig2.update_layout(
        height=380,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        yaxis=dict(gridcolor='#333333', range=[0, 1]),
        xaxis=dict(gridcolor='#333333'),
        yaxis_title="Average CAR"
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()


st.markdown("### 🔬 Sensor Readings — Current Zone")
s1, s2, s3, s4 = st.columns(4)

with s1:
    fig_co2 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest['co2_ppm'],
        title={'text': "CO₂ (ppm)"},
        gauge={
            'axis': {'range': [380, 1200]},
            'bar':  {'color': "red" if latest['co2_ppm'] > 700 else "green"},
            'steps': [
                {'range': [380, 500],  'color': 'rgba(0,255,0,0.2)'},
                {'range': [500, 700],  'color': 'rgba(255,255,0,0.2)'},
                {'range': [700, 1200], 'color': 'rgba(255,0,0,0.2)'}
            ]
        }
    ))
    fig_co2.update_layout(height=250,
                          paper_bgcolor='rgba(0,0,0,0)',
                          font=dict(color='white'))
    st.plotly_chart(fig_co2, use_container_width=True)

with s2:
    fig_moist = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest['soil_moisture'],
        title={'text': "Moisture (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar':  {'color': "blue"},
            'steps': [
                {'range': [0,  25],  'color': 'rgba(255,0,0,0.2)'},
                {'range': [25, 55],  'color': 'rgba(255,255,0,0.2)'},
                {'range': [55, 100], 'color': 'rgba(0,255,0,0.2)'}
            ]
        }
    ))
    fig_moist.update_layout(height=250,
                             paper_bgcolor='rgba(0,0,0,0)',
                             font=dict(color='white'))
    st.plotly_chart(fig_moist, use_container_width=True)

with s3:
    fig_temp = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest['temperature'],
        title={'text': "Temperature (°C)"},
        gauge={
            'axis': {'range': [0, 50]},
            'bar':  {'color': "orange"},
            'steps': [
                {'range': [0,  25], 'color': 'rgba(0,255,0,0.2)'},
                {'range': [25, 38], 'color': 'rgba(255,255,0,0.2)'},
                {'range': [38, 50], 'color': 'rgba(255,0,0,0.2)'}
            ]
        }
    ))
    fig_temp.update_layout(height=250,
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'))
    st.plotly_chart(fig_temp, use_container_width=True)

with s4:
    fig_ndvi = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest['ndvi'],
        title={'text': "NDVI"},
        gauge={
            'axis': {'range': [0, 1]},
            'bar':  {'color': "green"},
            'steps': [
                {'range': [0,   0.3], 'color': 'rgba(255,0,0,0.2)'},
                {'range': [0.3, 0.6], 'color': 'rgba(255,255,0,0.2)'},
                {'range': [0.6, 1.0], 'color': 'rgba(0,255,0,0.2)'}
            ]
        }
    ))
    fig_ndvi.update_layout(height=250,
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'))
    st.plotly_chart(fig_ndvi, use_container_width=True)

st.divider()


st.markdown("### 🗺️ Singrauli Corridor — Live Risk Heatmap")

zone_coords = {
    "Zone_1A": (24.1200, 82.6700),
    "Zone_2B": (24.1350, 82.6850),
    "Zone_3C": (24.1500, 82.7000),
    "Zone_4D": (24.1650, 82.7150),
    "Zone_5E": (24.1800, 82.7300),
}

def car_color(car):
    if car >= 0.65:   return 'green'
    elif car >= 0.40: return 'orange'
    elif car >= 0.25: return 'red'
    else:             return 'darkred'

m = folium.Map(
    location=[24.15, 82.71],
    zoom_start=12,
    tiles='CartoDB positron'
)

for zname, coords in zone_coords.items():
    cval  = zone_car.get(zname, 0.5)
    color = car_color(cval)
    risk, _, icon = get_risk(cval)

    folium.CircleMarker(
        location=coords,
        radius=35,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.6,
        popup=folium.Popup(
            f"<b>{zname}</b><br>"
            f"CAR: {cval:.3f}<br>"
            f"Risk: {risk}<br>"
            f"Recommendation: {get_recommendation(cval, zname)[:50]}...",
            max_width=250
        ),
        tooltip=f"{icon} {zname} — CAR: {cval:.3f} — {risk}"
    ).add_to(m)

    folium.Marker(
        location=coords,
        icon=folium.DivIcon(
            html=f'<div style="color:white;font-size:9px;'
                 f'font-weight:bold;text-align:center;">'
                 f'{zname}<br>CAR:{cval:.2f}</div>'
        )
    ).add_to(m)

for mine, data in legal_mines.items():
    folium.CircleMarker(
        location=(data['lat'], data['lon']),
        radius=15,
        color='yellow',
        fill=True,
        fill_color='yellow',
        fill_opacity=0.3,
        tooltip=f"⚠️ Legal Mine: {mine} ({data['operator']})",
        dash_array='5'
    ).add_to(m)

st_folium(m, height=450, use_container_width=True)
st.caption("🟢 Stable  🟡 Watch  🟠 Warning  🔴 Critical  🟡circle = Legal Mine Boundary")
st.divider()

if mode == "📊 Simulation Mode":
    st.markdown("### 🔄 Live CAR Simulation — Singrauli Corridor")
    st.info("Replaying 6 months of Singrauli sensor data in real time...")

    col_sim1, col_sim2 = st.columns(2)
    start_sim = col_sim1.button("▶️ Start Simulation", type="primary")
    speed     = col_sim2.slider("Speed", 0.1, 1.0, 0.3)

    if start_sim:
        placeholder = st.empty()
        chart_data  = []

        for i in range(0, min(200, len(zone_df))):
            row      = zone_df.iloc[i]
            X_live   = pd.DataFrame([row[features]])
            car_pred = xgb_model.predict(X_live)[0]
            risk, _, icon = get_risk(car_pred)
            chart_data.append({
                'time': row['timestamp'],
                'CAR':  car_pred
            })

            with placeholder.container():
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Live CAR",  f"{car_pred:.3f}")
                c2.metric("Risk",      f"{icon} {risk}")
                c3.metric("CO₂",       f"{row['co2_ppm']:.0f}")
                c4.metric("Moisture",  f"{row['soil_moisture']:.1f}%")
                c5.metric("Vibration", "YES" if row['vibration'] else "NO")

                if len(chart_data) > 5:
                    cdf      = pd.DataFrame(chart_data)
                    fig_live = px.line(cdf, x='time', y='CAR',
                                       color_discrete_sequence=['#00ff88'])
                    fig_live.add_hline(y=0.25, line_color="red",
                                       line_dash="dash")
                    fig_live.update_layout(
                        height=200,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        margin=dict(t=20, b=20)
                    )
                    st.plotly_chart(fig_live, use_container_width=True)

                if car_pred < 0.25:
                    st.error("🚨 CRITICAL ALERT FIRED!")
                    st.error(get_recommendation(car_pred, zone))

            time.sleep(speed)


elif mode == "📡 Live Sensor Mode":
    st.markdown("### 📡 Live ESP32 Sensor Feed")

    tab1, tab2 = st.tabs(["🔌 Auto (ESP32)", "✋ Manual Input"])

    with tab1:
        st.warning("Connect ESP32 via USB → Run backend first")
        st.code("uvicorn backend.app:app --reload --port 8000")

        if st.button("🔄 Fetch Live Reading"):
            try:
                import requests
                resp = requests.get("http://localhost:8000/sensor",
                                    timeout=3)
                data = resp.json()
                st.success("✅ Live reading received!")
                st.json(data)
            except:
                st.error("❌ Backend not running. Use Manual Input tab.")

    with tab2:
        st.markdown("#### Enter sensor values manually:")
        c1, c2 = st.columns(2)
        moisture     = c1.number_input("Soil Moisture %",  0.0, 100.0,  35.0)
        co2          = c1.number_input("CO₂ ppm",        300.0, 1200.0, 500.0)
        temp         = c2.number_input("Temperature °C",   10.0,  50.0,  30.0)
        humidity_val = c2.number_input("Humidity %",        0.0, 100.0,  60.0)
        vibration    = st.checkbox("Vibration Detected?")
        ndvi_val     = st.slider("NDVI", 0.0, 1.0, 0.4)

        if st.button("🔍 Predict CAR Now", type="primary"):
            X_input = pd.DataFrame([{
                'soil_moisture':   moisture,
                'co2_ppm':         co2,
                'temperature':     temp,
                'humidity':        humidity_val,
                'vibration':       int(vibration),
                'ndvi':            ndvi_val,
                'rainfall_mm':     0,
                'mining_activity': 5
            }])

            car_pred          = xgb_model.predict(X_input)[0]
            anomaly           = iso_model.predict(X_input)[0]
            is_anomaly        = anomaly == -1
            risk, rtype, icon = get_risk(car_pred)

            st.markdown("---")
            r1, r2, r3 = st.columns(3)
            r1.metric("Predicted CAR", f"{car_pred:.4f}")
            r2.metric("Risk Level",    f"{icon} {risk}")
            r3.metric("Anomaly",       "⚠️ YES" if is_anomaly else "✅ NO")

            if rtype == "error":
                st.error(f"🚨 CAR = {car_pred:.3f} — {risk}")
            elif rtype == "warning":
                st.warning(f"⚠️ CAR = {car_pred:.3f} — {risk}")
            else:
                st.success(f"✅ CAR = {car_pred:.3f} — {risk}")

            st.info(get_recommendation(car_pred, "Live Zone"))

            if is_anomaly:
                st.error("⚠️ ISOLATION FOREST: Anomalous pattern detected! "
                         "Possible illegal mining or sudden subsurface event!")


elif mode == "🕵️ Illegal Mining Demo":
    st.markdown("### 🕵️ Illegal Mining Detection Module")
    st.warning("This module cross-references CAR anomalies with legal mine database")

    st.markdown("#### Legal Mines Database — Singrauli")
    mines_df = pd.DataFrame([
        {"Mine": k,
         "Operator": v['operator'],
         "Lat": v['lat'],
         "Lon": v['lon'],
         "Approved Radius (km)": v['radius_km']}
        for k, v in legal_mines.items()
    ])
    st.dataframe(mines_df, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Simulate Illegal Mining Detection")

    col1, col2 = st.columns(2)
    alert_lat = col1.number_input("Alert Zone Latitude",  24.10, 24.20, 24.15)
    alert_lon = col2.number_input("Alert Zone Longitude", 82.65, 82.75, 82.71)
    car_drop  = st.slider("CAR Drop Amount", 0.1, 0.8, 0.4)
    vib_time  = st.selectbox("Vibration Time",
                              ["2AM", "3AM", "4AM", "10AM", "2PM"])
    night_vib = vib_time in ["2AM", "3AM", "4AM"]

    if st.button("🔍 Run Illegal Mining Check", type="primary"):
        from geopy.distance import geodesic

        alert_coords   = (alert_lat, alert_lon)
        nearest_mine   = None
        min_distance   = float('inf')
        nearest_radius = 0

        for mine, data in legal_mines.items():
            mine_coords = (data['lat'], data['lon'])
            distance    = geodesic(alert_coords, mine_coords).km
            if distance < min_distance:
                min_distance   = distance
                nearest_mine   = mine
                nearest_radius = data['radius_km']

        if min_distance <= nearest_radius:
            st.success("✅ LEGAL MINING ACTIVITY")
            st.info(f"Zone is within {nearest_mine} approved boundary "
                    f"({min_distance:.2f}km from mine center, "
                    f"radius {nearest_radius}km)")
        else:
            confidence = 0
            if car_drop > 0.3:   confidence += 40
            elif car_drop > 0.2: confidence += 25
            if night_vib:        confidence += 35
            if min_distance > 3: confidence += 25

            if min_distance < 1:   est_dist = "0-500m"
            elif min_distance < 2: est_dist = "500m-1km"
            elif min_distance < 3: est_dist = "1km-2km"
            else:                  est_dist = "2km+"

            st.error("🚨 ILLEGAL MINING DETECTED!")
            col_r1, col_r2, col_r3 = st.columns(3)
            col_r1.metric("Confidence",        f"{confidence}%")
            col_r2.metric("Estimated Distance", est_dist)
            col_r3.metric("Nearest Legal Mine", f"{min_distance:.2f}km away")

            st.error(
                f"⚠️ CLASSIFIED ALERT GENERATED\n\n"
                f"Suspected illegal mining activity detected "
                f"{est_dist} from sensor zone.\n"
                f"Nearest legal mine ({nearest_mine}) is "
                f"{min_distance:.2f}km away — outside approved "
                f"boundary of {nearest_radius}km.\n"
                f"Vibration at {vib_time} — "
                f"{'SUSPICIOUS (night hours)' if night_vib else 'Normal hours'}.\n\n"
                f"ACTION: Alert sent to Defense HQ + District Collector.\n"
                f"Confidence: {confidence}%"
            )

st.divider()


st.markdown("### 🚨 Alert History — All Zones")
alerts = df[df['alert'] == 1][
    ['timestamp', 'zone', 'CAR', 'risk_label', 'co2_ppm', 'soil_moisture']
].tail(15)
st.dataframe(alerts, use_container_width=True)

st.divider()


st.markdown(
    "**TERRA-SENSE** | CSIR-CMERI Defense Hackathon Track 2 | "
    "Singrauli Mining Corridor CAR Monitoring System | "
    f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import serial
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
import os
import io
from pymongo import MongoClient

# ── MONGODB ──────────────────────────────────────
client = MongoClient('mongodb://localhost:27017/')
db     = client['terrasense']
col    = db['sensor_readings']

app = FastAPI(title="TERRA-SENSE Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── MODELS ───────────────────────────────────────
xgb_model = pickle.load(open('ml/car_model.pkl',       'rb'))
iso_model  = pickle.load(open('ml/isolation_model.pkl', 'rb'))

# ── SQLITE (keep existing) ───────────────────────
def init_db():
    conn = sqlite3.connect('terrasense.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS readings (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT,
            zone        TEXT,
            co2_ppm     REAL,
            moisture    REAL,
            temperature REAL,
            humidity    REAL,
            vibration   INTEGER,
            CAR         REAL,
            risk        TEXT,
            alert       INTEGER,
            anomaly     INTEGER
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ── SERIAL ───────────────────────────────────────
def connect_serial():
    ports = ['COM12', 'COM4', 'COM5',
             '/dev/ttyUSB0', '/dev/ttyUSB1']
    for port in ports:
        try:
            ser = serial.Serial(port, 115200, timeout=2)
            print(f"ESP32 connected on {port}")
            return ser
        except:
            continue
    print("ESP32 not found — simulation mode")
    return None

ser = connect_serial()

FEATURES = [
    'soil_moisture', 'co2_ppm', 'temperature',
    'humidity', 'vibration', 'ndvi',
    'rainfall_mm', 'mining_activity'
]

# ── HELPERS ──────────────────────────────────────
def predict_car(sensor_data):
    X = pd.DataFrame([{
        'soil_moisture':   sensor_data.get('soil_moisture', 35),
        'co2_ppm':         sensor_data.get('co2_ppm', 500),
        'temperature':     sensor_data.get('temperature', 30),
        'humidity':        sensor_data.get('humidity', 60),
        'vibration':       sensor_data.get('vibration', 0),
        'ndvi':            sensor_data.get('ndvi', 0.4),
        'rainfall_mm':     sensor_data.get('rainfall_mm', 0),
        'mining_activity': sensor_data.get('mining_activity', 5)
    }])
    car     = float(np.clip(xgb_model.predict(X)[0], 0, 1))
    anomaly = int(iso_model.predict(X)[0] == -1)
    if car >= 0.65:   risk = "STABLE"
    elif car >= 0.40: risk = "WATCH"
    elif car >= 0.25: risk = "WARNING"
    else:             risk = "CRITICAL"
    return car, risk, anomaly

def dummy_sensor():
    import random
    return {
        "soil_moisture":   random.uniform(15, 45),
        "co2_ppm":         random.uniform(500, 800),
        "temperature":     random.uniform(28, 38),
        "humidity":        random.uniform(40, 70),
        "vibration":       random.choice([0, 1]),
        "ndvi":            random.uniform(0.1, 0.4),
        "rainfall_mm":     0,
        "mining_activity": random.uniform(5, 9)
    }

def get_recommendation(car):
    if car >= 0.65:   return "No action needed."
    elif car >= 0.40: return "Monitor closely. Begin afforestation."
    elif car >= 0.25: return "Plant Vetiver. Improve drainage. Backfill."
    else:             return "IMMEDIATE: Reroute convoys. Steel plates. Emergency Vetiver planting."

def clean_doc(d):
    """Convert MongoDB doc to JSON serializable"""
    if d:
        d['_id'] = str(d['_id'])
        if isinstance(d.get('timestamp'), datetime):
            d['timestamp'] = d['timestamp'].isoformat()
    return d

# ── EXISTING ENDPOINTS ───────────────────────────
@app.get("/")
def root():
    return {"system": "TERRA-SENSE", "status": "operational", "version": "1.0"}

@app.get("/sensor")
def get_sensor():
    if ser:
        try:
            line = ser.readline().decode('utf-8').strip()
            sensor_data = json.loads(line)
        except:
            sensor_data = dummy_sensor()
    else:
        sensor_data = dummy_sensor()

    car, risk, anomaly = predict_car(sensor_data)
    alert = 1 if car < 0.40 else 0
    now   = datetime.now()

    result = {
        "timestamp":      now.isoformat(),
        "zone":           "Zone_5E",
        "sensors":        sensor_data,
        "CAR":            round(car, 4),
        "risk":           risk,
        "alert":          bool(alert),
        "anomaly":        bool(anomaly),
        "recommendation": get_recommendation(car)
    }

    # ── SAVE TO SQLITE ──
    conn = sqlite3.connect('terrasense.db')
    conn.execute('''
        INSERT INTO readings
        (timestamp, zone, co2_ppm, moisture,
         temperature, humidity, vibration,
         CAR, risk, alert, anomaly)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    ''', (
        result['timestamp'], result['zone'],
        sensor_data.get('co2_ppm', 0),
        sensor_data.get('soil_moisture', 0),
        sensor_data.get('temperature', 0),
        sensor_data.get('humidity', 0),
        sensor_data.get('vibration', 0),
        car, risk, alert, anomaly
    ))
    conn.commit()
    conn.close()

    # ── SAVE TO MONGODB ──
    col.insert_one({
        'timestamp':     now,
        'zone':          result['zone'],
        'co2_ppm':       sensor_data.get('co2_ppm', 0),
        'soil_moisture': sensor_data.get('soil_moisture', 0),
        'temperature':   sensor_data.get('temperature', 0),
        'humidity':      sensor_data.get('humidity', 0),
        'vibration':     sensor_data.get('vibration', 0),
        'ndvi':          sensor_data.get('ndvi', 0),
        'rainfall_mm':   sensor_data.get('rainfall_mm', 0),
        'CAR':           round(car, 4),
        'risk':          risk,
        'alert':         bool(alert),
        'anomaly':       bool(anomaly),
        'recommendation':get_recommendation(car)
    })

    return result

@app.get("/history")
def get_history(limit: int = 50):
    conn = sqlite3.connect('terrasense.db')
    rows = conn.execute(
        f'SELECT * FROM readings ORDER BY id DESC LIMIT {limit}'
    ).fetchall()
    conn.close()
    return {"readings": rows}

@app.get("/zones")
def get_zones():
    df = pd.read_csv('data/singrauli_sensor_data.csv')
    zone_summary = df.groupby('zone').agg(
        avg_CAR=('CAR', 'mean'),
        min_CAR=('CAR', 'min'),
        alert_count=('alert', 'sum')
    ).reset_index()
    return zone_summary.to_dict(orient='records')

# ══════════════════════════════════════════
# NEW MONGODB ENDPOINTS
# ══════════════════════════════════════════

@app.get("/db/query")
def query_data(
    start: str = Query(None, description="ISO datetime e.g. 2024-01-01T00:00:00"),
    end:   str = Query(None, description="ISO datetime e.g. 2024-01-31T23:59:59"),
    zone:  str = Query(None, description="Zone_5E etc"),
    risk:  str = Query(None, description="CRITICAL/WARNING/WATCH/STABLE"),
    limit: int = Query(200)
):
    q = {}
    if start and end:
        q['timestamp'] = {
            '$gte': datetime.fromisoformat(start),
            '$lte': datetime.fromisoformat(end)
        }
    if zone: q['zone'] = zone
    if risk: q['risk'] = risk

    docs = list(col.find(q, {'_id':0})
                   .sort('timestamp', 1)
                   .limit(limit))
    for d in docs:
        if isinstance(d.get('timestamp'), datetime):
            d['timestamp'] = d['timestamp'].isoformat()

    return {'data': docs, 'count': len(docs)}


@app.get("/db/stats")
def get_stats(
    start: str = Query(None),
    end:   str = Query(None)
):
    q = {}
    if start and end:
        q['timestamp'] = {
            '$gte': datetime.fromisoformat(start),
            '$lte': datetime.fromisoformat(end)
        }

    pipeline = [
        {'$match': q},
        {'$group': {
            '_id':       None,
            'avg_car':   {'$avg': '$CAR'},
            'min_car':   {'$min': '$CAR'},
            'max_car':   {'$max': '$CAR'},
            'avg_co2':   {'$avg': '$co2_ppm'},
            'max_co2':   {'$max': '$co2_ppm'},
            'avg_moist': {'$avg': '$soil_moisture'},
            'avg_temp':  {'$avg': '$temperature'},
            'total':     {'$sum': 1},
            'criticals': {'$sum': {'$cond': [{'$eq': ['$risk','CRITICAL']}, 1, 0]}},
            'warnings':  {'$sum': {'$cond': [{'$eq': ['$risk','WARNING']},  1, 0]}},
            'anomalies': {'$sum': {'$cond': ['$anomaly', 1, 0]}}
        }}
    ]
    result = list(col.aggregate(pipeline))
    if result:
        r = result[0]
        r.pop('_id', None)
       
        for k,v in r.items():
            if isinstance(v, float):
                r[k] = round(v, 3)
        return r
    return {}


@app.get("/db/peaks")
def get_peaks():
    peak_co2    = col.find_one({}, {'_id':0}, sort=[('co2_ppm',    -1)])
    lowest_car  = col.find_one({}, {'_id':0}, sort=[('CAR',         1)])
    highest_car = col.find_one({}, {'_id':0}, sort=[('CAR',        -1)])
    last_crit   = col.find_one({'risk':'CRITICAL'}, {'_id':0}, sort=[('timestamp',-1)])
    last_anom   = col.find_one({'anomaly':True},    {'_id':0}, sort=[('timestamp',-1)])

    def fix(d):
        if d and isinstance(d.get('timestamp'), datetime):
            d['timestamp'] = d['timestamp'].isoformat()
        return d

    return {
        'peak_co2':     fix(peak_co2),
        'lowest_car':   fix(lowest_car),
        'highest_car':  fix(highest_car),
        'last_critical':fix(last_crit),
        'last_anomaly': fix(last_anom)
    }


@app.get("/db/trend")
def get_trend(
    start: str = Query(None),
    end:   str = Query(None),
    group_by: str = Query('hour', description="hour / day / week")
):
    q = {}
    if start and end:
        q['timestamp'] = {
            '$gte': datetime.fromisoformat(start),
            '$lte': datetime.fromisoformat(end)
        }

    
    if group_by == 'day':
        fmt = '%Y-%m-%d'
    elif group_by == 'week':
        fmt = '%Y-W%V'
    else:
        fmt = '%Y-%m-%dT%H'

    pipeline = [
        {'$match': q},
        {'$group': {
            '_id':      {'$dateToString': {'format': fmt, 'date': '$timestamp'}},
            'avg_car':  {'$avg': '$CAR'},
            'avg_co2':  {'$avg': '$co2_ppm'},
            'avg_moist':{'$avg': '$soil_moisture'},
            'avg_temp': {'$avg': '$temperature'},
            'count':    {'$sum': 1}
        }},
        {'$sort': {'_id': 1}}
    ]
    result = list(col.aggregate(pipeline))
    for r in result:
        for k,v in r.items():
            if isinstance(v, float):
                r[k] = round(v, 3)
    return {'trend': result}


@app.get("/db/export")
def export_excel(
    start: str = Query(None),
    end:   str = Query(None),
    zone:  str = Query(None),
    risk:  str = Query(None)
):
    q = {}
    if start and end:
        q['timestamp'] = {
            '$gte': datetime.fromisoformat(start),
            '$lte': datetime.fromisoformat(end)
        }
    if zone: q['zone'] = zone
    if risk: q['risk'] = risk

    docs = list(col.find(q, {'_id':0}).sort('timestamp',1))
    for d in docs:
        if isinstance(d.get('timestamp'), datetime):
            d['timestamp'] = d['timestamp'].isoformat()

    if not docs:
        return {"error": "No data found for given range"}

    df  = pd.DataFrame(docs)
    buf = io.BytesIO()

    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sensor Readings')

       
        summary = {
            'Total Readings':    len(df),
            'Average CAR':       round(df['CAR'].mean(), 3),
            'Min CAR':           round(df['CAR'].min(), 3),
            'Max CAR':           round(df['CAR'].max(), 3),
            'Average CO2 (PPM)': round(df['co2_ppm'].mean(), 1),
            'Max CO2 (PPM)':     round(df['co2_ppm'].max(), 1),
            'Critical Events':   len(df[df['risk']=='CRITICAL']),
            'Warning Events':    len(df[df['risk']=='WARNING']),
            'Anomalies':         len(df[df['anomaly']==True]),
            'Export Time':       datetime.now().isoformat()
        }
        pd.DataFrame([summary]).T.to_excel(
            writer, sheet_name='Summary', header=False
        )

    buf.seek(0)
    filename = f"terrasense_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

    return StreamingResponse(
        buf,
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        headers={'Content-Disposition': f'attachment; filename={filename}'}
    )


@app.get("/db/count")
def get_count():
    total    = col.count_documents({})
    critical = col.count_documents({'risk':'CRITICAL'})
    warning  = col.count_documents({'risk':'WARNING'})
    anomaly  = col.count_documents({'anomaly':True})
    return {
        'total':    total,
        'critical': critical,
        'warning':  warning,
        'anomaly':  anomaly
    }
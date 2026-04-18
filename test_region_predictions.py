import requests

regions = [
    ('Tokyo', 35.0, 139.0),
    ('London', 51.5, -0.1),
    ('New York', 40.7, -74.0),
    ('San Francisco', 37.77, -122.41),
    ('Sydney', -33.87, 151.21),
    ('Mexico City', 19.43, -99.13),
    ('Istanbul', 41.01, 28.98),
    ('Singapore', 1.35, 103.82),
    ('Los Angeles', 34.05, -118.24),
    ('Bangkok', 13.73, 100.51),
    ('Moscow', 55.75, 37.62),
    ('Rio de Janeiro', -22.91, -43.17),
    ('Seoul', 37.57, 126.98),
    ('Mumbai', 19.08, 72.88),
    ('Cairo', 30.04, 31.24),
]

print('=' * 90)
print('REGION MODEL PREDICTIONS')
print('=' * 90)
print('Region            | Tsunami (Mag 7.0) | Earthquake (Mag 6.0, Depth 100km)')
print('-' * 90)

for name, lat, lon in regions:
    ts_payload = {
        'Latitude': lat,
        'Longitude': lon,
        'Tsunami Magnitude (Iida)': 7.0,
        'Year': 2024,
        'Mo': 1,
        'Dy': 15,
        'Hr': 12,
        'Mn': 0,
        'Sec': 0,
    }
    eq_payload = {
        'Latitude': lat,
        'Longitude': lon,
        'Magnitude': 6.0,
        'Depth': 100,
        'Year': 2024,
        'Month': 1,
        'Day': 15,
        'Hour': 12,
        'Minute': 0,
        'Second': 0,
    }
    ts_resp = requests.post('http://127.0.0.1:5000/tsunami', json=ts_payload)
    eq_resp = requests.post('http://127.0.0.1:5000/earthquake', json=eq_payload)
    ts_pred = ts_resp.json().get('prediction')
    eq_pred = eq_resp.json().get('prediction')
    ts_status = 'HIGH' if ts_pred == 1 else 'LOW '
    eq_status = 'HIGH' if eq_pred == 1 else 'LOW '
    print(f'{name:16s} | {ts_status:4s}              | {eq_status:4s}')

print('=' * 90)

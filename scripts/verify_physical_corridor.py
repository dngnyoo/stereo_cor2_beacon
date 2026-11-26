import numpy as np

R_sun_km = 695700.0
r_sun_px_ref = 8.163
delta_t_ref = 1800.0

forward_px_ref = 30.0
backward_px_ref = 8.0

v_forward_ref = forward_px_ref * R_sun_km / (delta_t_ref * r_sun_px_ref)
v_backward_ref = backward_px_ref * R_sun_km / (delta_t_ref * r_sun_px_ref)

print(f"r_sun_px_ref: {r_sun_px_ref}")
print(f"v_forward_ref: {v_forward_ref:.2f} km/s")
print(f"v_backward_ref: {v_backward_ref:.2f} km/s")

# Under nominal conditions (1800s, 8.163 px/Rsun):
forward_calc = v_forward_ref * delta_t_ref * r_sun_px_ref / R_sun_km
backward_calc = v_backward_ref * delta_t_ref * r_sun_px_ref / R_sun_km

print(f"Reproduced forward_px: {forward_calc:.6f} (expected: 30.0)")
print(f"Reproduced backward_px: {backward_calc:.6f} (expected: 8.0)")

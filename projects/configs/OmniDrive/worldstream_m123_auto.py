"""The sole recommended route: ordinary M1 + admitted M2/M3 Auto/native fallback."""
_base_ = ['./worldstream_m2_window_auto.py']
model = dict(
    m1_config='configs/worldstream_m1_final.yaml',
    m3_enable=True,
    m3_config=dict(mode='auto', memory_budget_mb=512,
        cost_table='outputs/final_route/m3_cost_table.json',
        quality_certificate='outputs/final_route/m3_quality_certificate.json'))

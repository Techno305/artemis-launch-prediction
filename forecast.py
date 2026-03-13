import numpy as np
import pandas as pd
import datetime
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression


# -----------------------------------------
# PARAMETERS
# -----------------------------------------

SIMULATIONS = 20000
TARGET_DAYS = 420


# -----------------------------------------
# TRAINING DATA (schedule delay patterns)
# -----------------------------------------

data = {
    "days_before_launch":[800,700,600,500,400,300,200,150,120,90],
    "previous_delays":[0,1,1,2,2,3,3,4,4,5],
    "complexity":[5,5,6,7,7,8,8,9,9,10],
    "launched":[1,1,1,1,1,0,0,0,0,0]
}

df = pd.DataFrame(data)

X = df[["days_before_launch","previous_delays","complexity"]]
y = df["launched"]


# -----------------------------------------
# TRAIN MODEL
# -----------------------------------------

model = LogisticRegression()
model.fit(X,y)


# -----------------------------------------
# CURRENT ARTEMIS ESTIMATE
# -----------------------------------------

launch_features = [[150,3,9]]

prob_launch = model.predict_proba(launch_features)[0][1]

print("Probability Artemis II launches before 2026:", round(prob_launch,3))


# -----------------------------------------
# MONTE CARLO SIMULATION
# -----------------------------------------

today = datetime.date.today()

delay_distribution = np.random.normal(120,60,SIMULATIONS)

delay_distribution = np.maximum(delay_distribution,0)

launch_days = TARGET_DAYS + delay_distribution


# -----------------------------------------
# ESTIMATE DATE STATISTICS
# -----------------------------------------

median_days = np.median(launch_days)

p10 = np.percentile(launch_days,10)

p90 = np.percentile(launch_days,90)

median_date = today + datetime.timedelta(days=int(median_days))

p10_date = today + datetime.timedelta(days=int(p10))

p90_date = today + datetime.timedelta(days=int(p90))


print("\nEstimated Launch Date:", median_date)
print("Likely Window:", p10_date, "to", p90_date)


# -----------------------------------------
# BUILD PROBABILITY SURFACE
# -----------------------------------------

delay_axis = np.linspace(0,300,100)

time_axis = np.linspace(300,800,100)

X_mesh, Y_mesh = np.meshgrid(time_axis, delay_axis)

Z = np.exp(-(X_mesh - np.mean(launch_days))**2 / (2*np.std(launch_days)**2))


# -----------------------------------------
# INTERACTIVE 3D PLOT
# -----------------------------------------

fig = go.Figure(data=[go.Surface(
    x=X_mesh,
    y=Y_mesh,
    z=Z
)])

fig.update_layout(
    title="Artemis II Launch Probability Surface",
    scene=dict(
        xaxis_title="Days From Today",
        yaxis_title="Delay Magnitude",
        zaxis_title="Probability Density"
    ),
    width=900,
    height=700
)

fig.show()
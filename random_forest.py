
"""
Try a Random Forest to identify extremely metal-poor stars in SkyMapper.

0.  Photometrically identify stars with [Fe/H] < -3 with high completeness and 
    low contamination.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split

from itertools import combinations
from astropy.table import vstack

import utils

np.random.seed(42)

# Let's first try a Random Forest regressor to see if we can predict the value
# of the spectroscopic metallicity.

DATA_PATH = "data/"


emps_known = Table.read(
    os.path.join(DATA_PATH, "multiphot_known_EMP.final.dat"), format="ascii")
emps_observed = Table.read(
    os.path.join(DATA_PATH, "multiphot_observed_2.3m.final.dat"), format="ascii")
catalog = Table.read(os.path.join(DATA_PATH, "DR1_clean_RAVEon.fits"))

keep = catalog["QC"] * (catalog["TEFF"] > 4250) * (catalog["LOGG"] < 5) 
catalog = catalog[keep]
catalog["FeH"] = catalog["FE_H"]

#catalog = catalog[np.random.choice(len(catalog), 00)]


# Build X data array from all permutations of colours.
def prepare_arrays(catalogs, magnitude_labels, y_label=None, extra_label=None, dummy_value=99.99):

    Y = []
    X = []
    colors = []
    for i, j in combinations(range(len(magnitude_labels)), 2):
        x = []
        for catalog in catalogs:
            x.extend(catalog[magnitude_labels[i]] - catalog[magnitude_labels[j]])
        X.append(x)
        colors.append(" - ".join([magnitude_labels[i], magnitude_labels[j]]))

    X = np.vstack(X).T

    # Add second order colour terms?
    X = np.hstack([X, X**2])

    # Clean away problems with dummy values.
    bad = np.abs(X) > dummy_value/2.0
    for i in range(X.shape[1]):
        X[bad[:, i], i] = np.mean(X[~bad[:, i], i])

    if y_label is not None:
        for catalog in catalogs:
            Y.extend(catalog[y_label])
        Y = np.array(Y).reshape((-1, 1))

    else:
        Y = np.zeros((X.shape[0], 1))

    extras = []
    if extra_label is not None:
        for catalog in catalogs:
            if extra_label in catalog.dtype.names:
                extras.extend(catalog[extra_label])
            else:
                extras.extend(np.nan * np.ones(len(catalog)))
    else:
        extras = np.nan * np.ones(Y.size)
    extras = np.array(extras)
    extras[extras == dummy_value] = np.nan

    finite = np.isfinite(Y).flatten()
    X, Y, extras = (X[finite], Y[finite], extras[finite])

    return (X, Y, extras, colors)


magnitude_labels = ("u_mag", "v_mag", "g_mag", "r_mag", "i_mag")
X, y, C, colors = prepare_arrays((emps_known, emps_observed, catalog),
    magnitude_labels, "FeH", extra_label="CFe")

finite = np.isfinite(C)
C[~finite] = np.median(C[finite])
if not np.any(finite):
    C[:] = 1

X_train, X_test, y_train, y_test, C_train, C_test = train_test_split(X, y, C,
    train_size=0.95, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)


fig, ax = plt.subplots()
indices = np.argsort(C)#[:np.isfinite(C_test).sum()]
scat = ax.scatter(y[indices], model.predict(X)[indices], c=C[indices], 
    s=1, alpha=0.75)
plt.colorbar(scat)

utils.common_limits(ax, (-7, 1))


fig, ax = plt.subplots()
indices = np.argsort(C_test)#[:np.isfinite(C_test).sum()]
scat = ax.scatter(y_test[indices], model.predict(X_test)[indices], c=C_test[indices], 
    s=1, alpha=0.75)
plt.colorbar(scat)

utils.common_limits(ax, (-7, 1))



X_test_emps, y_test_emps, C_test_emps, colors = prepare_arrays(
    (emps_observed, ), magnitude_labels, "FeH")
fig, ax = plt.subplots()
scat = ax.scatter(y_test_emps, model.predict(X_test_emps), s=1, alpha=0.75)

utils.common_limits(ax, (-7, 1))


# Make predictions for DR1
dr1 = Table.read(os.path.join(DATA_PATH, "DR1_clean.dat"), format="ascii")

X_dr1, _, __, ___ = prepare_arrays((dr1, ), magnitude_labels)
dr1_feh = model.predict(X_dr1)


fig, ax = plt.subplots()
ax.hist(dr1_feh, log=True, edgecolor="k", bins=50)
ax.set_xlabel(r"$[{\rm Fe/H}]_{\rm predicted}$")
ax.set_ylabel("number of targets in SkyMapper DR1")

fig.tight_layout()
fig.savefig("dr1_rfr.png", dpi=300)


# Find new UMP candidates
known_emp_ids = set(list(emps_known["SkyMapper_ID"]) + list(emps_observed["SkyMapper_ID"]))
new_emp_ids = set(dr1["SkyMapper_ID"][(dr1_feh < -4)]).difference(known_emp_ids)

new_emp_indices = np.array([np.where(dr1["SkyMapper_ID"] == eid)[0][0] for eid in list(new_emp_ids)])



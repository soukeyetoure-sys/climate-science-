import os
import netCDF4 as nc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Chargement
ds    = nc.Dataset(os.path.expanduser("~/Téléchargements/CRU_Africa.nc"))
pre   = np.ma.filled(ds.variables['pre'][:], np.nan)
lon   = ds.variables['lon'][:]
lat   = ds.variables['lat'][:]
ds.close()

ntime, nlat, nlon = pre.shape
print(f"Shape: {ntime} mois, {nlat} lat, {nlon} lon")
months = np.tile(np.arange(1,13), ntime//12)

# 2. Reshape (pixels, temps) + masque
X          = pre.reshape(ntime, nlat*nlon).T
mask_valid = ~np.isnan(X).any(axis=1)
X_valid    = X[mask_valid]
print(f"Pixels valides: {mask_valid.sum()} / {nlat*nlon}")

# 3. Standardisation
X_scaled = StandardScaler().fit_transform(X_valid)

# 4. PCA
pca   = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)
expl  = pca.explained_variance_ratio_ * 100
print("Variance expliquee:", [f"{v:.1f}%" for v in expl])

# Figure 1 : variance expliquee
fig, ax = plt.subplots(figsize=(8,4))
ax.bar(range(1,11), expl, color='steelblue', alpha=0.8, label='Par composante')
ax.plot(range(1,11), np.cumsum(expl), 'ro-', lw=2, ms=6, label='Cumule')
ax.set_xlabel("Composante Principale")
ax.set_ylabel("Variance expliquee (%)")
ax.set_title("PCA - Variance Expliquee (CRU TS4.05)")
ax.legend(); ax.grid(alpha=0.4)
plt.tight_layout()
plt.savefig("pca_variance.png", dpi=150); plt.close()
print("OK pca_variance.png")

# Figures 2-3 : patrons spatiaux PC1 et PC2
for i in range(2):
    sp_map = np.full(nlat*nlon, np.nan)
    sp_map[mask_valid] = X_pca[:, i]
    sp_2d = sp_map.reshape(nlat, nlon)
    fig, ax = plt.subplots(figsize=(12,7))
    vmax = np.nanpercentile(np.abs(sp_2d), 98)
    cf = ax.contourf(lon, lat, sp_2d, levels=20,
                     cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(cf, ax=ax, label='Score PCA')
    ax.set_title(f"PC{i+1} - Patron Spatial ({expl[i]:.1f}% variance)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"pca_pc{i+1}_spatial.png", dpi=150); plt.close()
    print(f"OK pca_pc{i+1}_spatial.png")

# 5. Elbow
inertias = []
for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_pca)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(7,4))
ax.plot(range(2,10), inertias, 'bo-', lw=2, ms=7)
ax.set_xlabel("Nombre de clusters K")
ax.set_ylabel("Inertie")
ax.set_title("Methode Elbow - Choix du K optimal")
ax.grid(alpha=0.4)
plt.tight_layout()
plt.savefig("kmeans_elbow.png", dpi=150); plt.close()
print("OK kmeans_elbow.png")

# 6. K-Means K=5
K      = 5
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_pca)

c_map = np.full(nlat*nlon, np.nan)
c_map[mask_valid] = labels.astype(float)
c_2d  = c_map.reshape(nlat, nlon)

fig, ax = plt.subplots(figsize=(13,8))
cf = ax.contourf(lon, lat, c_2d,
                 levels=np.arange(-0.5, K+0.5),
                 cmap=plt.cm.get_cmap('tab10', K))
plt.colorbar(cf, ax=ax, ticks=range(K), label='Cluster')
ax.set_title(f"K-Means (K={K}) - Zones Climatiques Afrique (CRU TS4.05)")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("kmeans_clusters_map.png", dpi=150); plt.close()
print("OK kmeans_clusters_map.png")

# Cycle annuel par cluster
fig, ax = plt.subplots(figsize=(10,5))
colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple']
for k in range(K):
    idx   = np.where(labels == k)[0]
    ts    = X_valid[idx].mean(axis=0)
    cycle = [ts[months == m].mean() for m in range(1,13)]
    ax.plot(range(1,13), cycle, 'o-', lw=2,
            color=colors[k], label=f'Cluster {k+1}')

ax.set_xticks(range(1,13))
ax.set_xticklabels(['Jan','Fev','Mar','Avr','Mai','Jun',
                    'Jul','Aou','Sep','Oct','Nov','Dec'])
ax.set_xlabel("Mois"); ax.set_ylabel("Precipitation (mm/mois)")
ax.set_title("Cycle Annuel Moyen par Cluster - Afrique (CRU TS4.05)")
ax.legend(ncol=2); ax.grid(alpha=0.4)
plt.tight_layout()
plt.savefig("kmeans_annual_cycle_clusters.png", dpi=150); plt.close()
print("OK kmeans_annual_cycle_clusters.png")

print("\nTous les fichiers PNG dans :", os.getcwd())

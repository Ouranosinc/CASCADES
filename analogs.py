import numpy as np
import pandas as pd
import xarray as xr
import xscen as xs
import xclim.ensembles
import matplotlib.pyplot as plt

xs.load_config("project.yml", "paths.yml", "cfg.yml")

def main():

    pcat = xs.ProjectCatalog(xs.CONFIG["project"]["path"], create=True, project=xs.CONFIG["project"], overwrite=False)

    # to_dask() opens each separate dataset within a python dictionary
    hist_dict = pcat.search(source="CRCM5.*", processing_level=".*0.91vs.*").to_dataset_dict()
    hist = xclim.ensembles.create_ensemble(hist_dict)

    # to_dask() opens a dataset when the result of the search is a single dataset
    ref = pcat.search(source="ERA5.*", variable=["spei3", "spei6", "spei9"]).to_dask()
    # ClimEx was bias adjusted on a smaller domain. Use the same region.
    ref = ref.where(~np.isnan(hist["spei3"].isel(realization=0)).all(dim="time"))

    # Let's say that we want to find the SPEI-9 October
    target = ref["spei9"].sel(time="2021-10-01")
    candidates = hist["spei9"].where(hist.time.dt.month == 10, drop=True)

    # First screeening through RMSE. Only the events within the first 10 quantiles are kept
    rmse = np.sqrt(np.square(candidates - target).sum(dim=["lon", "lat"]))
    rmse = rmse.where(rmse != 0).compute()
    sorted_rmse = np.sort(rmse.values.flatten())
    candidates = candidates.where(rmse <= np.nanquantile(sorted_rmse, 0.10))

    # Pearson correlation
    corr_test = xr.corr(candidates, target, dim=["lon", "lat"]).compute()
    sorted_corr = pd.Series(corr_test.values.flatten()).sort_values(ascending=False)  # Numpy somehow has no sort in descending order?!?!

    # Plot
    plt.subplots(2, 4)
    ax = plt.subplot(1, 2, 1)
    target.plot(vmin=-3.09, vmax=3.09)
    plt.title("ERA5-Land | SPEI-9 | 2021-10-01")

    for i in range(4):
        ax = plt.subplot(2, 4, i + (3 if i <= 1 else 5))
        candidates.where(corr_test==sorted_corr.iloc[i], drop=True).squeeze().plot(vmin=-3.09, vmax=3.09)
        plt.title(f"{str(candidates.where(corr_test==sorted_corr.iloc[i], drop=True).squeeze().realization.values).split('.')[0].split('_')[4]} | "
                  f"{str(candidates.where(corr_test==sorted_corr.iloc[i], drop=True).squeeze().time.values).split(' ')[0]} | RMSE = {np.round(rmse.where(corr_test==sorted_corr.iloc[i], drop=True).squeeze().values, 1)}"
                  f" | Corr = {np.round(sorted_corr.iloc[i], 2)}")
    plt.tight_layout()

if __name__ == '__main__':
    main()

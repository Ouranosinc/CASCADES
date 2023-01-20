import numpy as np
import xarray as xr
import xscen as xs
import xclim.ensembles
import matplotlib.pyplot as plt

xs.load_config("project.yml", "paths.yml", "cfg.yml")

def main():

    pcat = xs.ProjectCatalog(xs.CONFIG["project"]["path"], create=True, project=xs.CONFIG["project"], overwrite=False)

    # to_dask() opens each separate dataset within a python dictionary
    hist_dict = pcat.search(source="CRCM5.*", processing_level=".*0.8.*").to_dataset_dict()
    # Temporary fix. I'll change that in the extraction function.
    for k, ds in hist_dict.items():
        hist_dict[k] = ds.clip(min=-3.09, max=3.09)
    hist = xclim.ensembles.create_ensemble(hist_dict)

    # to_dask() opens a dataset when the result of the search is a single dataset
    ref = pcat.search(source="ERA5.*", variable=["spei3", "spei6", "spei9"]).to_dask()
    # ClimEx was bias adjusted on a smaller domain. Use the same region.
    ref = ref.where(~np.isnan(hist_dict[k]["spei3"]).all(dim="time"))
    # Temporary fix. I'll change that in the extraction function.
    ref = ref.clip(min=-3.09, max=3.09)

    # Let's say that we want to find the SPEI-10 October
    target = ref["spei9"].sel(time="2021-10-01")
    candidates = hist["spei9"].where(hist.time.dt.month == 10, drop=True)
    corr_test = xr.corr(candidates, target, dim=["lon", "lat"]).compute()
    best = corr_test.argmax(...)

    # Plot
    plt.subplots(1, 2)
    ax = plt.subplot(1, 2, 1)
    target.plot(vmin=-3.09, vmax=3.09)
    plt.title("ERA5-Land | SPEI-9 | 2021-10-01")

    ax = plt.subplot(1, 2, 2)
    candidates.isel(realization=best["realization"], time=best["time"]).plot(vmin=-3.09, vmax=3.09)
    plt.title(f"{str(candidates.isel(realization=best['realization']).realization.values).split('.')[0]} | SPEI-9 | "
              f"{str(candidates.isel(time=best['time']).time.values).split(' ')[0]} | Corr = {np.round(corr_test.isel(realization=best['realization'], time=best['time']).values, 3)}")




if __name__ == '__main__':
    main()

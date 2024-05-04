# BANDS contain info about resolution as well as min and max wavelength for each band, useful for preprocessing of data
# DESCRIPTORS are used when passing data to a SEnSeI-enabled model

SENTINEL2_BANDS = [
            {"name": "B01", "resolution": 60, "band_type": "TOA Reflectance", "min_wavelength": 425.0, "max_wavelength": 461.0},
            {"name": "B02", "resolution": 10, "band_type": "TOA Reflectance", "min_wavelength": 446.0, "max_wavelength": 542.0},
            {"name": "B03", "resolution": 10, "band_type": "TOA Reflectance", "min_wavelength": 537.5, "max_wavelength": 582.5},
            {"name": "B04", "resolution": 10, "band_type": "TOA Reflectance", "min_wavelength": 645.5, "max_wavelength": 684.5},
            {"name": "B05", "resolution": 20, "band_type": "TOA Reflectance", "min_wavelength": 694.0, "max_wavelength": 714.0},
            {"name": "B06", "resolution": 20, "band_type": "TOA Reflectance", "min_wavelength": 731.0, "max_wavelength": 749.0},
            {"name": "B07", "resolution": 20, "band_type": "TOA Reflectance", "min_wavelength": 767.0, "max_wavelength": 795.0},
            {"name": "B08", "resolution": 10, "band_type": "TOA Reflectance", "min_wavelength": 763.5, "max_wavelength": 904.5},
            {"name": "B8A", "resolution": 20, "band_type": "TOA Reflectance", "min_wavelength": 847.5, "max_wavelength": 880.5},
            {"name": "B09", "resolution": 60, "band_type": "TOA Reflectance", "min_wavelength": 930.5, "max_wavelength": 957.5},
            {"name": "B10", "resolution": 60, "band_type": "TOA Reflectance", "min_wavelength": 1337.0, "max_wavelength": 1413.0},
            {"name": "B11", "resolution": 20, "band_type": "TOA Reflectance", "min_wavelength": 1541.0, "max_wavelength": 1683.0},
            {"name": "B12", "resolution": 20, "band_type": "TOA Reflectance", "min_wavelength": 2074.0, "max_wavelength": 2314.0},
        ]  

LANDSAT89_BANDS = [
            {"name": "B1", "resolution": 30, "band_type": "TOA Reflectance", "min_wavelength": 435.0, "max_wavelength": 451.0, 'GAIN': 0.00002, 'OFFSET': -0.1, 'SOLAR_CORRECTION': True},
            {"name": "B2", "resolution": 30, "band_type": "TOA Reflectance", "min_wavelength": 452.0, "max_wavelength": 512.0, 'GAIN': 0.00002, 'OFFSET': -0.1, 'SOLAR_CORRECTION': True},
            {"name": "B3", "resolution": 30, "band_type": "TOA Reflectance", "min_wavelength": 533.5, "max_wavelength": 590.5, 'GAIN': 0.00002, 'OFFSET': -0.1, 'SOLAR_CORRECTION': True},
            {"name": "B4", "resolution": 30, "band_type": "TOA Reflectance", "min_wavelength": 636.5, "max_wavelength": 673.5, 'GAIN': 0.00002, 'OFFSET': -0.1, 'SOLAR_CORRECTION': True},
            {"name": "B5", "resolution": 30, "band_type": "TOA Reflectance", "min_wavelength": 851.0, "max_wavelength": 879.0, 'GAIN': 0.00002, 'OFFSET': -0.1, 'SOLAR_CORRECTION': True},
            {"name": "B6", "resolution": 30, "band_type": "TOA Reflectance", "min_wavelength": 1566.5, "max_wavelength": 1651.5, 'GAIN': 0.00002, 'OFFSET': -0.1, 'SOLAR_CORRECTION': True},
            {"name": "B7", "resolution": 30, "band_type": "TOA Reflectance", "min_wavelength": 2114.5, "max_wavelength": 2287.5, 'GAIN': 0.00002, 'OFFSET': -0.1, 'SOLAR_CORRECTION': True},
            {"name": "B8", "resolution": 15, "band_type": "TOA Reflectance", "min_wavelength": 496.5, "max_wavelength": 683.5, 'GAIN': 0.00002, 'OFFSET': -0.1, 'SOLAR_CORRECTION': True},
            {"name": "B9", "resolution": 30, "band_type": "TOA Reflectance", "min_wavelength": 1363.5, "max_wavelength": 1384.5, 'GAIN': 0.00002, 'OFFSET': -0.1, 'SOLAR_CORRECTION': True},
            {"name": "B10", "resolution": 30, "band_type": "TOA Normalised Brightness Temperature", "min_wavelength": 10600.0, "max_wavelength": 11190.0, 'GAIN': 0.0003342, 'OFFSET': 0.1, 'K1': 774.8853, 'K2': 1321.0789, 'MINIMUM_BT': 132.0, 'MAXIMUM_BT': 249.0, 'SOLAR_CORRECTION': False},
            {"name": "B11", "resolution": 30, "band_type": "TOA Normalised Brightness Temperature", "min_wavelength": 11500.0, "max_wavelength": 12510.0, 'GAIN': 0.0003342, 'OFFSET': 0.1, 'K1': 480.8883, 'K2': 1201.1442, 'MINIMUM_BT': 127.0, 'MAXIMUM_BT': 239.0, 'SOLAR_CORRECTION': False},
        ]   

SENTINEL2_DESCRIPTORS = []
for band in SENTINEL2_BANDS:
    SENTINEL2_DESCRIPTORS.append({
                    'band_type': band['band_type'],
                    'min_wavelength': band['min_wavelength'],
                    'max_wavelength': band['max_wavelength']
    })


LANDSAT89_DESCRIPTORS = []
for band in LANDSAT89_BANDS:
    LANDSAT89_DESCRIPTORS.append({
                    'band_type': band['band_type'],
                    'min_wavelength': band['min_wavelength'],
                    'max_wavelength': band['max_wavelength']
    })
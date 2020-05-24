#from SREMPyLandsat.SREMPyLandsat import SREMPyLandsat
from L8LST_PSWA.LSTRetriever import LSTRetriever
from Landsat8SimpleAlbedo.AlbedoRetriever import AlbedoRetriever
from LandsatBasicUtils.MetadataReader import LandsatMetadataReader
from LandsatBasicUtils.BandCalibrator import LandsatBandCalibrator

import os, gdal, math
import numpy as np
from scipy.signal import convolve2d


class Landsat8AutoET():

    # LST Options
    cloudbufferdistance = 30
    cloudprobthreshold = 75.0
    shadowbufferdistance = 0
    NDVIs = 0.2
    NDVIv = 0.5

    lst_reflectance_correction_options = ['raw','dos','srem']
    albedo_reflectance_correction_options = ['raw','dos','srem','mixed_v1']
    albedo_method_options = ['olmedo','tasumi','beg','liang']
    lst_lse_mode_options = ['auto-ndvi-raw', 'auto-ndvi-dos', 'auto-ndvi-srem', 'from-glc', 'external']
    meteodata_type_options = ['ncep']

    def __init__(self, landsat_metadata_file, dem_file, meteodata_type, temp_dir):
        self.metadata_file = landsat_metadata_file
        self.metadata =LandsatMetadataReader(self.metadata_file)
        self.temp_dir = temp_dir
        self.dem_file = dem_file
        self.meteodata_type = meteodata_type
        if self.meteodata_type not in self.meteodata_type_options:
            raise ValueError('Unsupported meteodata type. Supported: %s' % self.meteodata_type_options)

    def setup_landsat_processing(self,
                                 lst_reflectance_correction='dos',
                                 lst_lse_mode='auto-ndvi-dos',
                                 lst_window_size=7,
                                 albedo_reflectance_correction='dos',
                                 albedo_method='olmedo',
                                 landsat_angles_file=None,
                                 usgs_utils_path=None,
                                 cygwin_bash_exe_path=None):

        self.lst_reflectance_correction = lst_reflectance_correction
        if not self.lst_reflectance_correction in self.lst_reflectance_correction_options:
            raise ValueError('Unsupported lst_reflectance_correction type. Supported: %s' % str(self.lst_reflectance_correction_options))

        self.lst_lse_mode = lst_lse_mode
        if not self.lst_lse_mode in self.lst_lse_mode_options:
            raise ValueError('Unsupported lst_lse_mode type. Supported: %s' % str(self.lst_lse_mode_options))

        self.lst_window_size = lst_window_size
        try:
            int(self.lst_window_size)
        except:
            raise ValueError('lst_window_size must be int')

        self.albedo_reflectance_correction = albedo_reflectance_correction
        if not self.albedo_reflectance_correction in self.albedo_reflectance_correction_options:
            raise ValueError('Unsupported albedo_reflectance_correction type. Supported: %s' % str(
                self.albedo_reflectance_correction_options))

        self.albedo_method = albedo_method
        if not self.albedo_method in self.albedo_method_options:
            raise ValueError('Unsupported albedo_method type. Supported: %s' % str(
                self.albedo_method_options))

        self.landsat_angles_file = landsat_angles_file
        self.usgs_utils_path = usgs_utils_path
        self.cygwin_bash_exe_path = cygwin_bash_exe_path

        pass


    def setup_meteo_processing(self,
                               ncep_autodownload=True,
                               ncep_file=None):

        # needed data:
        #
        pass

    def get_instant_ET_as_array(self):
        # 1. Prepare Landsat data
        # 1.1. Calculate LST, NDVI, Cloud mask, Water mask
        #lst_retriever = LSTRetriever(
        #    metadata_file=self.metadata_file,
        #    LSE_mode=self.lst_lse_mode,
        #    angles_file=self.landsat_angles_file,
        #    usgs_utils=self.usgs_utils_path,
        #    temp_dir=self.temp_dir,
        #    window_size=self.lst_window_size,
        #    cygwin_bash_exe_path=self.cygwin_bash_exe_path)
        #
        #lst = lst_retriever.get_lst_array()

        # temp
        lst_ds = gdal.Open(os.path.join(self.temp_dir,'latest_lst.tif'))
        lst = lst_ds.GetRasterBand(1).ReadAsArray()
        # end temp

        cloud_mask_ds = gdal.Open(os.path.join(self.temp_dir, 'latest_cloud.tif'))
        cloud_mask = cloud_mask_ds.GetRasterBand(1).ReadAsArray()
        # 1 - terrain
        # 2 - cloud
        # 3 - cloud shadow
        # 4 - snow
        # 5 - water

        ndvi_ds = gdal.Open(os.path.join(self.temp_dir,'latest_ndvi.tif'))
        ndvi = ndvi_ds.GetRasterBand(1).ReadAsArray()
        ndvi[np.isnan(lst)==True] = np.nan

        nir_ds = gdal.Open(os.path.join(self.temp_dir,'latest_b5_reflectance.tif'))
        nir = nir_ds.GetRasterBand(1).ReadAsArray()
        nir[np.isnan(lst) == True] = np.nan

        red_ds = gdal.Open(os.path.join(self.temp_dir, 'latest_b4_reflectance.tif'))
        red = red_ds.GetRasterBand(1).ReadAsArray()
        red[np.isnan(lst) == True] = np.nan

        # 1.2. Calculate albedo
        albedo_retriever = AlbedoRetriever(
            metadata_file=self.metadata_file,
            angles_file=self.landsat_angles_file,
            temp_dir=self.temp_dir,
            dem_file=self.dem_file,
            albedo_method=self.albedo_method,
            correction_method=self.albedo_reflectance_correction,
            usgs_utils=self.usgs_utils_path,
            cygwin_bash_exe_path=self.cygwin_bash_exe_path)

        albedo = albedo_retriever.get_albedo_as_array()
        albedo[np.isnan(lst) == True] = np.nan
        self.__save_array_to_gtiff(albedo, ndvi_ds, os.path.join(self.temp_dir,'latest_albedo.tif'))

        # 1.3. Calculate LAI
        print ('Calculating LAI')
        lai = self.calculate_lai(nir, red)

        # 1.4.
        print('Calculating broadband LAI')
        lse_broadband = 0.95 + 0.01 * lai
        lse_broadband[lai > 3] = 0.98
        lse_broadband[np.isnan(lai)==True] = np.nan
        self.__save_array_to_gtiff(lse_broadband, ndvi_ds, os.path.join(self.temp_dir, 'latest_lse_broadband.tif'))

        # check
        print ('LST: %s' % str(lst.shape))
        print('NDVI: %s' % str(ndvi.shape))
        print('LAI: %s' % str(lai.shape))
        print('Cloud: %s' % str(cloud_mask.shape))
        print('Albedo: %s' % str(albedo.shape))
        print('LSE: %s' % str(lse_broadband.shape))

        # 2. Init DEM
        dem_ds = gdal.Open(self.dem_file)
        geoTransform = ndvi_ds.GetGeoTransform()
        projection = ndvi_ds.GetProjection()
        xMin = geoTransform[0]
        yMax = geoTransform[3]
        xMax = xMin + geoTransform[1] * ndvi_ds.RasterXSize
        yMin = yMax + geoTransform[5] * ndvi_ds.RasterYSize
        xSize = ndvi_ds.RasterXSize
        ySize = ndvi_ds.RasterYSize
        print('Recalculating DEM to scene domain')
        dem_ds_converted = gdal.Warp('', dem_ds, format='MEM', outputBounds=(xMin, yMin, xMax, yMax),
                                     width=xSize, height=ySize, dstSRS=projection)
        dem = dem_ds_converted.GetRasterBand(1).ReadAsArray()
        self.__save_array_to_gtiff(dem, ndvi_ds, os.path.join(self.temp_dir, 'latest_dem.tif'))

        # 3. Anchor pixels selection
        #  [Bhattarai2017] Nishan Bhattarai, Lindi J. Quackenbush, Jungho Im,
        #         Stephen B. Shaw, 2017.
        #         A new optimized algorithm for automating endmember pixel selection
        #         in the SEBAL and METRIC models.
        #         Remote Sensing of Environment, Volume 196, Pages 178-192,
        #         https://doi.org/10.1016/j.rse.2017.05.009.
        # exclude water
        ndvi_for_anchor = ndvi + 0.0
        ndvi_for_anchor[cloud_mask==5] = np.nan

        lst_for_anchor = lst + 0.0
        lst_for_anchor[cloud_mask == 5] = np.nan

        albedo_for_anchor = albedo + 0.0
        albedo_for_anchor[cloud_mask == 5] = np.nan

        # 3.1. Spatial homogenity calculation
        print ('Computing spatial homogenity metrics')
        print ('For NDVI...')
        cv_ndvi, _, _ = self.moving_cv_filter(ndvi_for_anchor, (11, 11))
        print('For LST...')
        cv_lst, _, std_lst = self.moving_cv_filter(lst_for_anchor, (11, 11))
        print('For Albedo...')
        cv_albedo, _, _ = self.moving_cv_filter(albedo_for_anchor, (11, 11))

        # 3.2. Iterative searching process
        print ('Searching for cold and hot pixels')
        cold, hot = self.esa(ndvi_for_anchor, lst_for_anchor, cv_ndvi, std_lst, cv_albedo)
        print ('Cold')
        print (cold)
        print ('Hot')
        print (hot)

        # 4. Aux data
        earth_sun_distance = float(self.metadata.metadata['EARTH_SUN_DISTANCE'])
        sun_elevation_angle = float(self.metadata.metadata['SUN_ELEVATION'])
        solar_incidence_angle = 90 - sun_elevation_angle
        solar_incidence_angle = math.radians(solar_incidence_angle)
        atmospheric_transmissivity = 0.75 + 0.00002*dem
        air_emissivity = 0.85 * ((-1*np.log(atmospheric_transmissivity))**0.09)

        # 5. incoming shortwave radiation (Rsi) - W/m2
        RSi = 1367 * math.cos(solar_incidence_angle)*(1 / (earth_sun_distance**2)) * atmospheric_transmissivity
        self.__save_array_to_gtiff(RSi, ndvi_ds, os.path.join(self.temp_dir, 'latest_RSi.tif'))

        # 6. outgoing longwave radiation (RLo) - W/m2
        RLo = lse_broadband * (lst**4) * 0.0000000567
        self.__save_array_to_gtiff(RLo, ndvi_ds, os.path.join(self.temp_dir, 'latest_RLo.tif'))

        # 7. Incoming longwave radiation (RLi) - W/m2
        RLi = air_emissivity * 0.0000000567 * (lst[cold]**4)
        self.__save_array_to_gtiff(RLi, ndvi_ds, os.path.join(self.temp_dir, 'latest_RLi.tif'))

        # 8. net radiation flux (Rn) - W/m2
        Rn = (1-albedo)*RSi + RLi - RLo - (1-lse_broadband)*RLi
        self.__save_array_to_gtiff(Rn, ndvi_ds, os.path.join(self.temp_dir, 'latest_Rn.tif'))

        # 9. Soil heat flux (W/m2)
        # Bastiaanssen, W.G.M.; Pelgrum, H.; Wang, J.; Ma, J.; Moreno, J.; Roerink, G.J.; Van Der Wal, T. (1998)  The  Surface  Energy  Balance  Algorithm  for  Land  (SEBAL):  Part  2  validation. J.  Hydrol. 1998, 228, 213-229
        G_Rn = ((lst-273.15)/albedo) * (0.0038*albedo + 0.0074*(albedo**2))* (1-(0.98*(ndvi**4)))
        G_Rn[ndvi<0] = 0.5
        G = Rn * G_Rn
        self.__save_array_to_gtiff(G, ndvi_ds, os.path.join(self.temp_dir, 'latest_G.tif'))



    ######### service functions ##########
    def calculate_lai(self, nir, red):
        # Bastiaanssen W. G. M. et al. Remote sensing in water resources management: The state of the art. – International Water Management Institute. – 1998.
        savi = self.calculate_savi(nir,red)
        lai = np.log((0.69 - savi) / 0.59) / 0.91
        lai[savi >= 0.687] = 6.0
        lai[np.isnan(savi)==True] = np.nan
        return lai

    def calculate_savi(self, nir, red):
        # Waters, R., Allen, R., Bastiaanssen, W., Tasumi, M., and Trezza, S. R. 2002. “Surface Energy Balance Algorithms for Land. Idaho Implementation.” Advanced Training and Users Manual, Idaho, USA.
        L = 0.5
        savi = ((1 + L) * (nir - red)) / (L + nir + red)
        return savi

    # Anchor pixels searching is from https://github.com/hectornieto/pyMETRIC
    # Many thanks to Hector Nieto (hector.nieto@irta.cat hector.nieto.solana@gmail.com)

    def moving_cv_filter(self, data, window):

        ''' window is a 2 element tuple with the moving window dimensions (rows, columns)'''
        kernel = np.ones(window) / np.prod(np.asarray(window))
        mean = convolve2d(data, kernel, mode='same', boundary='symm')

        distance = (data - mean) ** 2

        std = np.sqrt(convolve2d(distance, kernel, mode='same', boundary='symm'))

        cv = std / mean

        return cv, mean, std

    def histogram_fiter(self, vi_array, lst_array):
        cold_bin_pixels = 0
        hot_bin_pixels = 0
        bare_bin_pixels = 0
        full_bin_pixels = 0

        while (cold_bin_pixels < 50
               or hot_bin_pixels < 50
               or bare_bin_pixels < 50
               or full_bin_pixels < 50):

            max_lst = np.amax(lst_array)
            min_lst = np.amin(lst_array)
            max_vi = np.amax(vi_array)
            min_vi = np.amin(vi_array)

            print('Setting LST boundaries %s - %s' % (min_lst, max_lst))
            n_bins = int(np.ceil((max_lst - min_lst) / 0.25))
            lst_hist, lst_edges = np.histogram(lst_array, n_bins)

            print('Setting VI boundaries %s - %s' % (min_vi, max_vi))
            n_bins = int(np.ceil((max_vi - min_vi) / 0.01))
            vi_hist, vi_edges = np.histogram(vi_array, n_bins)

            # Get number of elements in the minimum and maximum bin
            cold_bin_pixels = lst_hist[0]
            hot_bin_pixels = lst_hist[-1]
            bare_bin_pixels = vi_hist[0]
            full_bin_pixels = vi_hist[-1]

            # Remove possible outliers
            if cold_bin_pixels < 50:
                lst_array = lst_array[lst_array >= lst_edges[1]]

            if hot_bin_pixels < 50:
                lst_array = lst_array[lst_array <= lst_edges[-2]]

            if bare_bin_pixels < 50:
                vi_array = vi_array[vi_array >= vi_edges[1]]

            if full_bin_pixels < 50:
                vi_array = vi_array[vi_array <= vi_edges[-2]]

        return lst_edges[0], lst_edges[-1], vi_edges[0], vi_edges[-1]

    def rank_array(self, array):

        temp = array.argsort(axis=None)
        ranks = np.arange(np.size(array))[temp.argsort()].reshape(array.shape)

        return ranks

    def incremental_search(self, vi_array, lst_array, mask, is_cold=True):
        step = 0
        if is_cold:
            while True:

                for n_lst in range(1, 11 + step):
                    for n_vi in range(1, 11 + step):
                        print('Searching cold pixels from the %s %% minimum LST and %s %% maximum VI' % (n_lst, n_vi))
                        vi_high = np.percentile(vi_array[mask], 100 - n_vi)
                        lst_cold = np.percentile(lst_array[mask], n_lst)
                        cold_index = np.logical_and.reduce((mask,
                                                            vi_array >= vi_high,
                                                            lst_array <= lst_cold))

                        if np.sum(cold_index) >= 10:
                            return cold_index

                # If we reach here is because not enought pixels were found
                # Incresa the range of percentiles
                step += 5
                if step > 90:
                    return []
        else:
            while True:
                for n_lst in range(1, 11 + step):
                    for n_vi in range(1, 11 + step):
                        print('Searching hot pixels from the %s %% maximum LST and %s %% minimum VI' % (n_lst, n_vi))
                        vi_low = np.percentile(vi_array[mask], n_vi)
                        lst_hot = np.percentile(lst_array[mask], 100 - n_lst)
                        hot_index = np.logical_and.reduce((mask,
                                                           vi_array <= vi_low,
                                                           lst_array >= lst_hot))

                        if np.sum(hot_index) >= 10:
                            return hot_index
                # If we reach here is because not enought pixels were found
                # Incresa the range of percentiles
                step += 5
                if step > 90:
                    return []

    def esa(self,
            vi_array,
            lst_array,
            cv_vi,
            std_lst,
            cv_albedo):
        ''' Finds the hot and cold pixel using the
        Exhaustive Search Algorithm

        Parameters
        ----------
        vi_array : numpy array
            Vegetation Index array (-)
        lst_array : numpy array
            Land Surface Temperature array (Kelvin)
        cv_ndvi : numpy array
            Coefficient of variation of Vegetation Index as homogeneity measurement
            from neighboring pixels
        std_lst : numpy array
            Standard deviation of LST as homogeneity measurement
            from neighboring pixels
        cv_albedo : numpy array
            Coefficient of variation of albdeo as homogeneity measurement
            from neighboring pixels
        Returns
        -------
        cold_pixel : int or tuple

        hot_pixel : int or tuple
        ETrF_cold : float

        ETrF_hot : float

        References
        ----------
        .. [Bhattarai2017] Nishan Bhattarai, Lindi J. Quackenbush, Jungho Im,
            Stephen B. Shaw, 2017.
            A new optimized algorithm for automating endmember pixel selection
            in the SEBAL and METRIC models.
            Remote Sensing of Environment, Volume 196, Pages 178-192,
            https://doi.org/10.1016/j.rse.2017.05.009.
        '''

        lst_nan = np.isnan(lst_array)
        vi_nan = np.isnan(vi_array)
        if np.all(lst_nan) or np.all(vi_nan):
            print('No valid LST or VI pixels')
            return None, None

        # Step 1. Find homogeneous pixels
        print('Filtering pixels by homgeneity')
        homogeneous = np.logical_and.reduce((cv_vi <= 0.25,
                                             cv_albedo <= 0.25,
                                             std_lst < 1.5))

        print('Found %s homogeneous pixels' % np.sum(homogeneous))
        if np.sum(homogeneous) == 0:
            return None, None

        # Step 2 Filter outliers by Building ndvi and lst histograms
        lst_min, lst_max, vi_min, vi_max = self.histogram_fiter(vi_array[~vi_nan],
                                                           lst_array[~lst_nan])

        print('Removing outliers by histogram')
        mask = np.logical_and.reduce((homogeneous,
                                      lst_array >= lst_min,
                                      lst_array <= lst_max,
                                      vi_array >= vi_min,
                                      vi_array <= vi_max))

        print('Keep %s pixels after outlier removal' % np.sum(mask))
        if np.sum(mask) == 0:
            return None, None

        # Step 3. Interative search of cold pixel
        print('Iterative search of candidate cold pixels')
        cold_pixels = self.incremental_search(vi_array, lst_array, mask, is_cold=True)
        print('Found %s candidate cold pixels' % np.sum(cold_pixels))
        if np.sum(cold_pixels) == 0:
            return None, None

        print('Iterative search of candidate hot pixels')
        hot_pixels = self.incremental_search(vi_array, lst_array, mask, is_cold=False)
        print('Found %s candidate hot pixels' % np.sum(hot_pixels))
        if np.sum(hot_pixels) == 0:
            return None, None

        # Step 4. Rank the pixel candidates
        print('Ranking candidate anchor pixels')
        lst_rank = self.rank_array(lst_array)
        vi_rank = self.rank_array(vi_array)
        rank = vi_rank - lst_rank
        cold_pixel = np.logical_and(cold_pixels, rank == np.max(rank[cold_pixels]))

        cold_pixel = tuple(np.argwhere(cold_pixel)[0])
        print('Cold  pixel found with %s K and %s VI' % (float(lst_array[cold_pixel]),
                                                         float(vi_array[cold_pixel])))

        rank = lst_rank - vi_rank
        hot_pixel = np.logical_and(hot_pixels, rank == np.max(rank[hot_pixels]))

        hot_pixel = tuple(np.argwhere(hot_pixel)[0])
        print('Hot  pixel found with %s K and %s VI' % (float(lst_array[hot_pixel]),
                                                        float(vi_array[hot_pixel])))

        return cold_pixel, hot_pixel

    def __save_array_to_gtiff(self, array, domain_raster, gtiff_path):
        driver = gdal.GetDriverByName("GTiff")
        dataType = gdal.GDT_Float32
        dataset = driver.Create(gtiff_path, array.shape[1], array.shape[0], domain_raster.RasterCount, dataType)
        dataset.SetProjection(domain_raster.GetProjection())
        dataset.SetGeoTransform(domain_raster.GetGeoTransform())

        dataset.GetRasterBand(1).WriteArray(array)

        del dataset


a = Landsat8AutoET(landsat_metadata_file='F:/LC08_L1TP_182019_20190830_20190903_01_T1/LC08_L1TP_182019_20190830_20190903_01_T1_MTL.txt',
                   dem_file = 'F:/LC08_L1TP_182019_20190830_20190903_01_T1/DEM.tif',
                   meteodata_type='ncep',
                   temp_dir='F:/LC08_L1TP_182019_20190830_20190903_01_T1/temp')

a.setup_landsat_processing()
a.get_instant_ET_as_array()
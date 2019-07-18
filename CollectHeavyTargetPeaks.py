from uimfpy.UIMFReader import *
import numpy as np

import matplotlib.pyplot as plt

from scipy.sparse import csc_matrix,csr_matrix,coo_matrix
import pickle

from scipy import ndimage as ndi
from skimage.feature import peak_local_max

from scipy.spatial.distance import cosine


class PeakRegion(object):
    """a rectangle region for a single peak"""
    def __init__(self, frame_start, frame_end, scan_start, scan_end):
        super(PeakRegion, self).__init__()
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.scan_start = scan_start
        self.scan_end = scan_end

def find_mzbin_infos(mzbins_by_mz, sequence, mono_mz, isotope, charge):
    for info in mzbins_by_mz:
        seq,_mz,mz,iso,z = info
        if (seq==sequence) & (z==charge) & (iso==isotope):
            return info
    return (sequence,mono_mz,mono_mz+isotope/charge, isotope, charge)

def find_heavy_mzbin_infos(mzbins_by_mz):
    infos = []
    for info in mzbins_by_mz:
        seq,_mz,mz,iso,z = info
        if (seq.endswith(']')) & (iso==0):
            infos.append(info)
    return infos

def find_light_mzbin_infos(mzbins_by_mz):
    infos = []
    for info in mzbins_by_mz:
        seq,_mz,mz,iso,z = info
        if (not seq.endswith(']')) & (iso==0):
            infos.append(info)
    return infos

def find_light_mzbin_infos_by_heavy(mzbins_by_mz, heavy_seq, heavy_z):
    light_seq = heavy_seq.rsplit('[+', 1)[0]
    infos = []
    for info in mzbins_by_mz:
        seq,_mz,mz,iso,z = info
        if (light_seq==seq) & (heavy_z==z):
            infos.append(info)
    return infos

def find_light_mono_info_by_heavy(mzbins_by_mz, heavy_seq, heavy_z):
    light_seq = heavy_seq.rsplit('[+', 1)[0]
    infos = []
    for info in mzbins_by_mz:
        seq,_mz,mz,iso,z = info
        if (light_seq==seq) & (heavy_z==z) & (iso==0):
            return info
    return None

def lc_isotope_cosine(reader, xic_by_mzbins, mzbins_by_mz, peak_region, seq, mz, z, isotopes=3):
    mono_lc = None
    lc_peaks_list = []
    lc_isotope_infos = dict()
    
    for iso in range(-1, isotopes+1):
        mzbins = mzbins_by_mz[find_mzbin_infos(mzbins_by_mz, seq, mz, iso, z)]
        xic2d = xic_matrix(xic_by_mzbins, mzbins, reader.num_frames, reader.num_scans, normalize=False)
        sub_xic2d = xic2d[peak_region.frame_start:peak_region.frame_end,peak_region.scan_start:peak_region.scan_end]
        lc_peaks = sub_xic2d.sum(axis=1)
        if iso == 0: mono_lc = lc_peaks
        lc_peaks_list.append(lc_peaks)
        
    for iso in range(-1, isotopes+1):
        idx = iso + 1
        area = np.sum(lc_peaks_list[idx])
        lc_isotope_infos['M{}_area'.format(iso)] = area
        if iso == 0 or area == 0: continue
        _lc_cosine_sim = 1-cosine(mono_lc, lc_peaks_list[idx])
        lc_isotope_infos['M{}_cosine'.format(iso)] = _lc_cosine_sim

    #     plt.plot(lc_peaks[peak_region.frame_start:peak_region.frame_end], label='[M+{}]'.format(iso))
    # plt.legend()
    # plt.show()
    return lc_isotope_infos

def read_target_list(target_file, sheet_name=0):
    '''read a target list file
        TODO: check out the format
    '''
    target_df = pd.read_excel(target_file, sheet_name=sheet_name)
    return target_df

def collect_target_mzbins(target_df, mz_calibrator_by_params, ppm=10, isotope=3, heavy_only=False):
    ''' collect target bin ranges for all mz_calibrators
    '''
    target_mzbins = dict()
    mzbins_by_mz = dict()
    for param in mz_calibrator_by_params:
        print(param)
        target_mzbins[param] = []
        mz_calibrator = mz_calibrator_by_params[param]
        for idx,t in target_df.iterrows():
            # @TODO
            # print(t['Modified Sequence'],t['Precursor Mz'],t['Precursor Charge'])
            seq,_mz,z = t['Modified Sequence'],t['Precursor Mz'],t['Precursor Charge']
            if heavy_only:
                if t['Isotope Label Type'] == 'light':
                    continue
            
            for iso in range(-1, isotope+1):
                mz = _mz + iso/z
                _max = mz*(1+ppm*1e-6)
                _min = mz*(1-ppm*1e-6)
                _minbin = int(mz_calibrator.MZtoBin(_min))+1
                _maxbin = int(mz_calibrator.MZtoBin(_max))+1

                l = list(range(_minbin,_maxbin))
                if len(l)==0:
                    # print(mz, 'no mz_bins due to the low resolution')
                    continue
                else:
                    target_mzbins[param] += l
                    mzbins_by_mz[(seq,_mz,mz,iso,z)] = l
        target_mzbins[param] = set(target_mzbins[param])
        print(param, ', #mzbins:', len(target_mzbins[param]))
    return target_mzbins,mzbins_by_mz

def lc_chromatogram(xic_by_mzbin, mzbins, n_frames, n_scans, fout=None):
    xic = csr_matrix((n_frames+1, n_scans+1), dtype=np.uint32)
    for mzbin in mzbins:
        if mzbin in xic_by_mzbin:
            xic += xic_by_mzbin[mzbin].tocsr()
    chrom_by_frame = xic.sum(axis=1)
    plt.close('all')
    plt.plot(chrom_by_frame)
    if fout is not None: plt.savefig(fout)
    plt.show()
    
def drift_chromatogram(xic_by_mzbin, mzbins, n_frames, n_scans, fout=None):
    xic = csc_matrix((n_frames+1, n_scans+1), dtype=np.uint32)
    for mzbin in mzbins:
        if mzbin in xic_by_mzbin:
            xic += xic_by_mzbin[mzbin].tocsc()
    chrom_by_scan = xic.sum(axis=0)
    plt.close('all')
    plt.plot(chrom_by_scan.reshape((-1, 1)))
    if fout is not None: plt.savefig(fout)
    plt.show()

def xic_matrix(xic_by_mzbin, mzbins, n_frames, n_scans, normalize=True):
    xic = csr_matrix((n_frames+1, n_scans+1), dtype=np.uint32)
    for mzbin in mzbins:
        if mzbin in xic_by_mzbin:
            xic += xic_by_mzbin[mzbin].tocsr()
    if normalize: return (xic / xic.max()).toarray()
    else: return xic.toarray()

    
def peaks_with_static_box(image2d, scale_factor, peaks, frame_offset, scan_offset, frames_for_peak, scans_for_peak):
    '''
        image2d: the local 2d xic (gray-scaled: 0-1)
        scale_factor: max intensity of the global 2d xic (scale factor)
        peaks: apexs in this local 2d xic
        frame_offset: frame offset index in the global 2d xic for the local 2d
        scan_offset: scan offset index in the global 2d xic for the local 2d
        frames_for_peak: fixed size of frames (retention time)
        frames_for_peak: fixed size of scans (drift time)
    '''
    local_peaks = []
    
    height, width = image2d.shape
    
    for k in range(peaks.shape[0]):
        min_h = peaks[k,0]-(frames_for_peak//2)
        max_h = peaks[k,0]+(frames_for_peak//2)
        min_w = peaks[k,1]-(scans_for_peak//2)
        max_w = peaks[k,1]+(scans_for_peak//2)
        
        # skip if the peak candidate is located near the edges
        if (min_h < 0) | (max_h >= height) | (min_w < 0) | (max_w >= width): continue
            
        local_peak = image2d[min_h:max_h,min_w:max_w]
        local_peaks.append({'frame':peaks[k,0]+frame_offset, 'scan':peaks[k,1]+scan_offset,
                            'frame_start':min_h+frame_offset, 'frame_end':max_h+frame_offset,
                            'scan_start':min_w+scan_offset, 'scan_end':max_w+scan_offset,
                            'peak_area':np.sum(local_peak)*scale_factor
                           })
    return local_peaks
            
    
def get_local_peaks(xic_2d, coordinates, frame_pad=50, scan_pad=100, npeaks_per_candidate=3,
                    masking_threshold=0.1, frames_for_peak=None, scans_for_peak=None, fout=None):
    xic_2d_max = xic_2d.max()
    im_gray = xic_2d/xic_2d_max
    max_frames,max_scans = xic_2d.shape

    local_peaks = []
    
    for i,j in coordinates:
        
        # extract a local 2d xic
        _min_frame = max(0,i-frame_pad)
        _max_frame = min(i+frame_pad,max_frames)
        _min_scan = max(0,j-scan_pad)
        _max_scan = min(j+scan_pad,max_scans)
        
        #print('Frames: %d:%d, Scans:%d:%d' %(_min_frame,_max_frame,_min_scan,_max_scan))
        
        image = im_gray[_min_frame:_max_frame,_min_scan:_max_scan]
        
        # filter by moving average
        moving_avg_image = ndi.uniform_filter(image, size=10, mode='constant')
        # filter by maximum (only for visualization)
        image_max = ndi.maximum_filter(image, size=10, mode='constant')
        # find peak candidates in moving_avg_image
        candidates = peak_local_max(moving_avg_image, min_distance=10, num_peaks=npeaks_per_candidate)
        
        if frames_for_peak is None or scans_for_peak is None:
            ##########################################################
            # find a rectangle bounding box for each peak candidate
            ##########################################################
            # make a binary masking with a threshold
            mask = moving_avg_image > masking_threshold
            # label the regions
            label_im, nb_labels = ndi.label(mask)
            sizes = ndi.sum(mask, label_im, range(nb_labels + 1))
            peak_area = ndi.sum(image, label_im, range(nb_labels + 1))

            #print(sizes)
            #print(peak_area)

            target_labels = []
            for k in range(candidates.shape[0]):
                # label of a candidate
                target_label = label_im[candidates[k,0], candidates[k,1]]
                print(candidates[k], target_label, sizes[target_label], peak_area[target_label])
                # if a region is meaningful
                if (sizes[target_label] > 0) & (target_label not in target_labels):
                    target_labels.append(target_label)
                    slice_x, slice_y = ndi.find_objects(label_im==target_label)[0]
                    #print('boundary:', slice_x, slice_y)

                    # save peaks
                    local_peaks.append({'frame':candidates[k,0]+_min_frame, 'scan':candidates[k,1]+_min_scan,
                                        'frame_start':slice_x.start+_min_frame, 'frame_end':slice_x.stop+_min_frame,
                                        'scan_start':slice_y.start+_min_scan, 'scan_end':slice_y.stop+_min_scan,
                                        'peak_area':peak_area[target_label]*xic_2d_max
                                       })
                    ##########################################################
                    # plot
                    ##########################################################
                    #roi = image[slice_x, slice_y]

                    #plt.figure(figsize=(4, 2))
                    #plt.axes([0, 0, 1, 1])
                    #plt.imshow(roi)
                    #plt.axis('off')

                    #plt.show()
            ##########################################################
            # find a rectangle bounding box for each peak candidate
            ##########################################################
        else:
            _local = peaks_with_static_box(image, xic_2d_max, candidates, _min_frame, _min_scan, frames_for_peak, scans_for_peak)
            local_peaks += _local
        
#         # display results
#         fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
#         ax = axes.ravel()
#         ax[0].imshow(image)
#         ax[0].axis('off')
#         ax[0].set_title('Original')

#         ax[1].imshow(image_max, cmap=plt.cm.gray)
#         ax[1].axis('off')
#         ax[1].set_title('Maximum filter')

#         ax[2].imshow(image, cmap=plt.cm.gray)
#         ax[2].autoscale(False)
#         ax[2].plot(candidates[:, 1], candidates[:, 0], 'r.')
#         ax[2].axis('off')
#         ax[2].set_title('Peak local max')

#         plt.tight_layout()
#         # plt.savefig('peak_local_max_{0:.2f}.png'.format(mz))
#         plt.show()
    return local_peaks

def collect_peaks(xic_by_mzbin, mzbins, num_frames, num_scans, num_top_peaks=20, 
                  num_global_peaks=20, frames_for_peak=None, scans_for_peak=None):
    im = xic_matrix(xic_by_mzbin, mzbins, num_frames, num_scans, normalize=False)
    #print(im.max())
    im_gray = im/im.max()

    moving_avg_image = ndi.uniform_filter(im_gray, size=10, mode='constant')

    # Comparison between image_max and im to find the coordinates of local maxima
    coordinates = peak_local_max(moving_avg_image, min_distance=20, num_peaks=num_global_peaks)

#     # display results
#     fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
#     ax = axes.ravel()
#     ax[0].imshow(im_gray)

#     ax[0].axis('off')
#     ax[0].set_title('Original')

#     ax[1].imshow(moving_avg_image, cmap=plt.cm.gray)
#     ax[1].axis('off')
#     ax[1].set_title('Maximum filter')

#     ax[2].imshow(im_gray, cmap=plt.cm.gray)
#     ax[2].autoscale(False)
#     ax[2].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
#     ax[2].axis('off')
#     ax[2].set_title('Peak local max')

#     plt.tight_layout()

#     plt.show()

#     lc_chromatogram(xic_by_mzbin, mzbins, reader.num_frames, reader.num_scans)
#     drift_chromatogram(xic_by_mzbin, mzbins, reader.num_frames, reader.num_scans)
    
    # local_peaks = get_local_peaks(im, coordinates, frame_pad=50, scan_pad=100, npeaks_per_candidate=3, masking_threshold=0.01)
    local_peaks = get_local_peaks(im, coordinates, frame_pad=50, scan_pad=100, npeaks_per_candidate=3,
                                  frames_for_peak=frames_for_peak, scans_for_peak=scans_for_peak)
    if len(local_peaks) == 0: return []
    local_peaks_df = pd.DataFrame(local_peaks).drop_duplicates()
    local_peaks_df = local_peaks_df.astype(np.uint32)
    return local_peaks_df.sort_values('peak_area', ascending=False).head(num_top_peaks).to_dict(orient='record')

def get_mz_peaks(reader, frame_arr, scan_arr, int_arr, mzbin_ranges_by_params, xic_by_mzbin, th=50):
    for f, s, i in zip(frame_arr, scan_arr, int_arr):
        params = reader.calibration_params_by_frame[f]
        slope, intercept = params['CalibrationSlope'], params['CalibrationIntercept']
        mz_calibrator = reader.mz_calibrator_by_params[(slope, intercept, reader.bin_width)]
        target_mzbin = target_mzbins[(slope, intercept, reader.bin_width)]
        mzbin_ranges = mzbin_ranges_by_params[(slope, intercept, reader.bin_width)]
        
        bin_intensities = decode(decompress(i))
        for idx, intensity in bin_intensities:
            if intensity > th:
                if  mzbin_ranges[0] <= idx <= mzbin_ranges[1]:
                    if idx not in xic_by_mzbin: xic_by_mzbin[idx] = []
                    xic_by_mzbin[idx].append([f,s,intensity])

def get_signals_in_peak_region(reader, peak_region, mz, z, isotope, ppm, sig_th=0):
    stime = time.time()
    xic_by_mzbin = dict()
    
    mzbin_ranges_by_params = reader.get_mzbin_ranges_by_mz_window(mz-1/z, mz+isotope/z, ppm=ppm)
    
    df = reader.read_frame_scans(frame_nums=list(range(peak_region.frame_start,peak_region.frame_end)),
                                 scan_nums=list(range(peak_region.scan_start,peak_region.scan_end)))
    
    # print(mzbin_ranges_by_params)
    get_mz_peaks(reader, df.FrameNum.values,
                 df.ScanNum.values,
                 df.Intensities.values,
                 mzbin_ranges_by_params,
                 xic_by_mzbin,
                 sig_th)
    # print("Frame:[{0}-{1}], Scan:[{2}-{3}] Done: Time: {4:.3f} s".format(
    #     peak_region.frame_start,
    #     peak_region.frame_end,
    #     peak_region.scan_start,
    #     peak_region.scan_end,
    #     time.time()-stime))
    
    for mzbin in xic_by_mzbin:
        _l = xic_by_mzbin[mzbin]
        arr = np.array(_l)
        xic_by_mzbin[mzbin] = coo_matrix((arr[:,2], (arr[:,0],arr[:,1])), shape=(reader.num_frames+1, reader.num_scans+1))
        _l = None
        arr = None
    
    return xic_by_mzbin

def collect_target_peaks(xic_by_mzbin, mzbins_by_mz, target_infos, num_frames, num_scans,
                         num_top_peaks=20, num_global_peaks=20,
                         frames_for_peak=20, scans_for_peak=20):
    '''collect the peak candidates based on the 2d xic
        xic_by_mzbin: 2d xic. it needs to be built from UIMF file
        mzbins_by_mz: mzbins by target infos
        target_infos: list of target infos
    '''
    cache_peaks_by_mzbins = dict()
    target_peaks = []
    n_targets = len(target_infos)
    print('# Targets:', n_targets)
    stime = time.time()
    
    for idx, info in enumerate(target_infos):
        seq,_mz,mz,iso,z = info
        target_mzbin = mzbins_by_mz[info]
        
        mzbins_key = ' '.join([str(m) for m in target_mzbin])
        # if there is a cache
        if mzbins_key in cache_peaks_by_mzbins:
            local_peaks = cache_peaks_by_mzbins[mzbins_key]
        else:
            local_peaks = collect_peaks(xic_by_mzbin, target_mzbin, num_frames, num_scans,
                                        num_top_peaks=num_top_peaks, num_global_peaks=num_global_peaks,
                                        frames_for_peak=frames_for_peak, scans_for_peak=scans_for_peak)
            cache_peaks_by_mzbins[mzbins_key] = local_peaks
        
        if len(local_peaks) > 0:
            df = pd.DataFrame(local_peaks)
            df['pepseq'] = seq
            df['mono_mass'] = _mz
            df['iso_mass'] = mz
            df['iso'] = iso
            df['z'] = z
            target_peaks.append(df)
        if (idx+1)%500 == 0: print("[%d/%d] %.2f min" % (idx+1, n_targets, (time.time()-stime)/60))
    
    rdf = pd.concat(target_peaks, ignore_index=True)
    
    print("[%d/%d] (%d peaks) Dnoe: %.2f min" % (idx+1, n_targets, rdf.shape[0], (time.time()-stime)/60))
    return rdf

def collect_heavy_targets(uimf, target_list_file, xic2d_file, fout=None):
    print("#"*150)
    print("# Collect Heavy Targets")
    print("# UIMF:", uimf)
    print("# Target List:", target_list_file)
    print("# 2D XIC:", xic2d_file)
    print("#"*150)
    
    stime = time.time()
    reader = UIMFReader(uimf, TIC_threshold=0)
    
    # read the target lists
    target_df = read_target_list(target_list_file)
    target_mzbins, mzbins_by_mz = collect_target_mzbins(target_df, reader.mz_calibrator_by_params, ppm=40, isotope=3, heavy_only=False)
    
    with open(xic2d_file, 'rb') as handle:
        xic_by_mzbin = pickle.load(handle)
    print('# ready xic_by_mzbin file,', (time.time()-stime)/60, 'min')
    
    heavy_target_infos = find_heavy_mzbin_infos(mzbins_by_mz)
    heavy_peaks = collect_target_peaks(xic_by_mzbin, mzbins_by_mz, heavy_target_infos, reader.num_frames, reader.num_scans,
                                       num_top_peaks=20, num_global_peaks=20,
                                       frames_for_peak=20, scans_for_peak=20)
    print('# collect_target_peaks, Done: ', (time.time()-stime)/60, 'min')
    
    if fout: heavy_peaks.to_csv(fout)
    
    return heavy_peaks


def get_heavy_light_lc_isotope_infos(heavy_info, reader, mzbins_by_mz, target_xic_by_mzbins=None):
    seq, mz, z = heavy_info['pepseq'], heavy_info['mono_mass'], heavy_info['z']
    peak_region = PeakRegion(heavy_info['frame_start'], heavy_info['frame_end'], heavy_info['scan_start'], heavy_info['scan_end'])

    # print(seq, mz, z)

    if target_xic_by_mzbins is None:
        xic_by_mzbins_in_local = get_signals_in_peak_region(reader, peak_region, mz=mz, z=z, isotope=3, ppm=40)
        heavy_lc_isotope_infos = lc_isotope_cosine(reader, xic_by_mzbins_in_local, mzbins_by_mz, peak_region, seq, mz, z, isotopes=3)
    else:
        heavy_lc_isotope_infos = lc_isotope_cosine(reader, target_xic_by_mzbins, mzbins_by_mz, peak_region, seq, mz, z, isotopes=3)

    # print(_i, idx)
    # print(heavy_lc_isotope_infos)

    light_info = find_light_mono_info_by_heavy(mzbins_by_mz, seq, z)
    # print(light_info)
    light_seq,light_mono_mz,_,light_iso,_ = light_info

    if target_xic_by_mzbins is None:
        xic_by_mzbins_in_local = get_signals_in_peak_region(reader, peak_region, mz=light_mono_mz, z=z, isotope=3, ppm=40)
        light_lc_isotope_infos = lc_isotope_cosine(reader, xic_by_mzbins_in_local, mzbins_by_mz, peak_region, light_seq, light_mono_mz, z, isotopes=3)
    else:
        light_lc_isotope_infos = lc_isotope_cosine(reader, target_xic_by_mzbins, mzbins_by_mz, peak_region, light_seq, light_mono_mz, z, isotopes=3)
    
    return heavy_lc_isotope_infos, light_lc_isotope_infos

####################################################################################################

data_folder = 'data/'

uimf = data_folder+'LC_{0}_heavy_repA_1_MinInt5_DComp5.uimf'.format(5000)
target_list_file = data_folder+'20190319-Slim charge1-5 light-heavy.xlsx'
xic2d_file = data_folder+'LC_{0}_heavy_repA_1_MinInt5_DComp5_50ppm.pkl'.format(5000)
peak_out_file = data_folder+'LC_{0}_heavy_repA_1_MinInt5_DComp5_50ppm_heavy_peaks.csv'.format(5000)
peak_evaluation_file = data_folder+'LC_{0}_heavy_repA_1_MinInt5_DComp5_50ppm_heavy_peaks_eval.pkl'.format(5000)

heavy_peaks_for_5000 = pd.read_csv(peak_out_file, index_col=0)
print(heavy_peaks_for_5000.head())
print(heavy_peaks_for_5000[heavy_peaks_for_5000.peak_area>1000].shape)
print(heavy_peaks_for_5000[heavy_peaks_for_5000.peak_area>2000].shape)
print(heavy_peaks_for_5000[heavy_peaks_for_5000.peak_area>5000].shape)

heavy_peaks_for_5000 = heavy_peaks_for_5000[heavy_peaks_for_5000.peak_area>1000]
n_heavy_peaks = heavy_peaks_for_5000.shape[0]

print('n_heavy_peaks:', n_heavy_peaks)
####################################################################################################

peak_evaluations_per_um = dict()
for _i in [0, 78, 156, 312, 625, 1250, 2500, 5000]:
    print("#"*100)
    stime = time.time()
    uimf = data_folder+'LC_{0}_heavy_repA_1_MinInt5_DComp5.uimf'.format(_i)
    reader = UIMFReader(uimf, TIC_threshold=0)
    print('# ready uimf: {},'.format(uimf), (time.time()-stime)/60, 'min')

    target_df = read_target_list(data_folder+'20190319-Slim charge1-5 light-heavy.xlsx', sheet_name=0)
    print('# ready targets:{0},'.format(target_df.shape[0]), (time.time()-stime)/60, 'min')

    target_mzbins, mzbins_by_mz = collect_target_mzbins(target_df, reader.mz_calibrator_by_params, ppm=40, isotope=3, heavy_only=False)
    print('# ready target m/z bins,', (time.time()-stime)/60, 'min')

    target_xic_by_mzbins = None
    target_xic2d_file = data_folder+'LC_{0}_heavy_repA_1_MinInt5_DComp5_50ppm.pkl'.format(_i)
    try:
        with open(target_xic2d_file, 'rb') as handle:
            target_xic_by_mzbins = pickle.load(handle)
    except FileNotFoundError:
        print("[ERR] There is no target_xic2d_file: %s" % target_xic2d_file)
    print('# ready target_xic2d file,', (time.time()-stime)/60, 'min')

    for idx in range(n_heavy_peaks):
        heavy_info = heavy_peaks_for_5000.iloc[idx]
        heavy_lc_isotope_infos, light_lc_isotope_infos = get_heavy_light_lc_isotope_infos(heavy_info, reader, mzbins_by_mz, target_xic_by_mzbins=target_xic_by_mzbins)

        peak_evaluations_per_um[(_i, idx, 'heavy')] = heavy_lc_isotope_infos
        peak_evaluations_per_um[(_i, idx, 'light')] = light_lc_isotope_infos

        if (idx+1)%100 == 0:
            print("# %d [%d/%d] %.2f min" % (_i, idx+1, n_heavy_peaks, (time.time()-stime)/60))
            # break
    print("# %d [%d/%d] Done %.2f min" % (_i, idx+1, n_heavy_peaks, (time.time()-stime)/60))

with open(peak_evaluation_file, 'wb') as f:
    pickle.dump(peak_evaluations_per_um, f, pickle.HIGHEST_PROTOCOL)

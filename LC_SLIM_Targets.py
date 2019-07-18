# import matplotlib
# matplotlib.use("TkAgg")
    
# import matplotlib.pyplot as plt
# from matplotlib.transforms import Affine2D
# from matplotlib.colors import LogNorm
# import matplotlib.patches as patches

from uimfpy.UIMFReader import *
import numpy as np

from scipy.sparse import csc_matrix,csr_matrix,coo_matrix
import pickle
import pandas as pd

from scipy import ndimage as ndi
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

from skimage.feature import peak_local_max
from skimage import data, img_as_float
from skimage.morphology import watershed
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb


import argparse

################################################
parser = argparse.ArgumentParser(description='test', formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument(
        '--uimf', type=str,
        help='uimf data file')
parser.add_argument(
        '--target', type=str,
        help='target list file')
parser.add_argument(
        '--bin_file', type=str,
        help='to save a target mz bin file')
parser.add_argument(
        '--ppm', type=float, default=20.0,
        help='ppm error')
parser.add_argument(
        '--rt', type=float, default=1.0,
        help='time window for the retention time (min)')
parser.add_argument(
        '--dt', type=float, default=100.0,
        help='time window for the drift time (ms)')
parser.add_argument(
        '--isotope', type=int, default=3,
        help='time window for the drift time')
parser.add_argument(
        '--sig_th', type=float, default=25,
        help='for collecting mzbins, a threshold for a single signal intensity')
parser.add_argument(
        '--mode', type=str, default='binning',
        help='binning: collect the mz bins for targets \
              peak: find peaks \
        ')
parser.add_argument(
        '--target_sheet_number', type=int, default=0,
        help='sheet number in a target excel file')
FLAGS = parser.parse_args()

print("-"*40)
print("Parameters")
print(FLAGS)
print("-"*40)
################################################


def read_target_list(target_file, sheet_name=0):
    '''read a target list file
        TODO: check out the format
    '''
    target_df = pd.read_excel(target_file, sheet_name=sheet_name)
    return target_df


def collect_target_mzbins(target_df, mz_calibrator_by_params, ppm=10, isotope=3):
    ''' collect target bin ranges for all mz_calibrators
    '''
    
    target_mzbins = dict()
    mzbins_by_mz = dict()
    for param in mz_calibrator_by_params:
        target_mzbins[param] = []
        mz_calibrator = mz_calibrator_by_params[param]
        for idx,t in target_df.iterrows():
            # @TODO
            # print(t['Modified Sequence'],t['Precursor Mz'],t['Precursor Charge'])
            seq,_mz,z = t['Modified Sequence'],t['Precursor Mz'],t['Precursor Charge']
            for iso in range(isotope+1):
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

import gc


def mz_binning_task(info):
    frame, scan, intensities = info
    bin_intensities = decode(decompress(intensities))
    arr = np.empty((len(bin_intensities), 4), dtype=np.uint32)
    arr[:,0:2] = np.array(bin_intensities, dtype=np.uint32)
    arr[:,2] = frame
    arr[:,3] = scan
    return arr

import multiprocessing

def get_mz_peaks_multi(frame_arr, scan_arr, int_arr, target_mzbin, xic_by_mzbin, th=50, processes=4):
    stime = time.time()
    pool = multiprocessing.Pool(processes=processes)
    results = pool.map(mz_binning_task, zip(frame_arr, scan_arr, int_arr))
    print("mz_binning for nrows:{0}, Done: Time: {1:.3f} s".format(
        len(results), time.time()-stime))
    arr = np.vstack(results)
    arr = arr[(arr[:,1] > th)&(np.isin(arr[:,0], target_mzbin))]
    for idx in range(arr.shape[0]):
        binidx, intensity, f, s = arr[idx,:]
        if binidx not in xic_by_mzbin: xic_by_mzbin[binidx] = []
        xic_by_mzbin[binidx].append([f,s,intensity])
    print("mz_binning for nrows:{0}, Done: Time: {1:.3f} s".format(
        arr.shape[0], time.time()-stime))

def get_mz_peaks(frame_arr, scan_arr, int_arr, target_mzbin, xic_by_mzbin, th=50):
    for f, s, i in zip(frame_arr, scan_arr, int_arr):
        bin_intensities = decode(decompress(i))
        for idx, intensity in bin_intensities:
            if intensity > th:
                if idx in target_mzbin:
                    if idx not in xic_by_mzbin: xic_by_mzbin[idx] = []
                    xic_by_mzbin[idx].append([f,s,intensity])

# def get_mz_peaks(reader, frame_arr, scan_arr, int_arr, target_mzbin, xic_by_mzbin, th=50):
#     results = []
#     for f, s, i in zip(frame_arr, scan_arr, int_arr):
#         bin_intensities = decode(decompress(i))
#         _arr = np.empty((len(bin_intensities), 4), dtype=np.uint32)
#         _arr[:,0:2] = np.array(bin_intensities, dtype=np.uint32)
#         _arr[:,2] = f
#         _arr[:,3] = s
#         results.append(_arr)

#     arr = np.vstack(results)
#     arr = arr[(arr[:,1] > th)&(np.isin(arr[:,0], target_mzbin))]
#     for idx in range(arr.shape[0]):
#         binidx, intensity, f, s = arr[idx,:]
#         if binidx not in xic_by_mzbin: xic_by_mzbin[binidx] = []
#         xic_by_mzbin[binidx].append([f,s,intensity])

def get_target_peaks(reader, target_mzbins, sig_th=50, ofile=None):
    stime = time.time()
    xic_by_mzbin = dict()
    chunk_size = 100
    num_chunks = reader.num_frames//chunk_size

    #### TODO: 
    params = reader.calibration_params_by_frame[1]
    slope, intercept = params['CalibrationSlope'], params['CalibrationIntercept']
    target_mzbin = target_mzbins[(slope, intercept, reader.bin_width)]
    #print('target_mzbin:', target_mzbin)

    for i in range(num_chunks):
        gc.disable()
        start = i*chunk_size + 1
        end = start + chunk_size
        
        df = reader.read_frame_scans(frame_nums=range(start,end))
        get_mz_peaks(df.FrameNum.values,
                     df.ScanNum.values,
                     df.Intensities.values,
                     target_mzbin,
                     xic_by_mzbin,
                     sig_th)
        print("[{0}-{1}] Done: Time: {2:.3f} s".format(
            start,
            end,
            time.time()-stime))
        gc.enable()

    for mzbin in xic_by_mzbin:
        _l = xic_by_mzbin[mzbin]
        arr = np.array(_l)
        xic_by_mzbin[mzbin] = coo_matrix((arr[:,2], (arr[:,0],arr[:,1])), shape=(reader.num_frames+1, reader.num_scans+1), dtype=np.uint32)
        _l = None
        arr = None
    
    # write a 2D XIC binned by m/z into pickle
    if ofile is not None: write_pickle(xic_by_mzbin, ofile)
    
    return xic_by_mzbin

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


def get_roi(xic_by_mzbin, mzbins, num_frames, num_scans, min_distance=5, num_peaks=5, normalize=False):
    im = xic_matrix(xic_by_mzbin, mzbins, num_frames, num_scans, normalize=normalize)
    im_gray = im/im.max()
    
    coordinates = peak_local_max(im_gray, min_distance=min_distance, num_peaks=num_peaks)
    return im, coordinates


# Median absolute deviation
def mad(x, axis=None):
    return np.median(np.abs(x - np.median(x, axis)), axis)


def peak_candidates(xic_2d, coordinates, frame_pad=50, scan_pad=100, npeaks=3, fout=None):
    xic_2d_max = xic_2d.max()
    im_gray = xic_2d/xic_2d_max
    #thresh = threshold_otsu(im_gray)
    #print('thresh:',thresh)
    
    local_peaks = []
    max_frames,max_scans = xic_2d.shape
    
    for i,j in coordinates:
        _min_frame = max(0,i-frame_pad)
        _max_frame = min(i+frame_pad,max_frames)
        _min_scan = max(0,j-scan_pad)
        _max_scan = min(j+scan_pad,max_scans)
        
        image = im_gray[_min_frame:_max_frame,_min_scan:_max_scan]
        
#         plt.close('all')
#         detected_peaks = detect_peaks(image)
#         plt.subplot(1,2,1)
#         plt.imshow(image)
#         plt.subplot(1,2,2)
#         plt.imshow(detected_peaks)
#         plt.show()
        
        # apply threshold @TODO
        bw = closing(image > 0, square(3))

        # remove artifacts connected to image border
        cleared = clear_border(bw)

        # label image regions
        label_image = label(cleared)
        #print(np.unique(label_image))
        
        peak_areas = []
        background_mad = 0
        for l in range(label_image.max()+1):
            idxs = np.where(label_image==l)
            peak_areas.append(xic_2d_max * np.sum(image[idxs]))
            if l==0:
                background_mad = mad(xic_2d_max*image[idxs]) # background
                background_std = np.std(xic_2d_max*image[idxs],ddof=1) # background
        sorted_idx = np.argsort(peak_areas)
        
        print('median absolute deviation in background area:', background_mad)
        print('standard deviation in background area:', background_std)
        
        areas = np.bincount(label_image.flatten())
#         print('areas:', areas)
#         sorted_idx = np.argsort(areas)
        #print('index:', sorted_idx[::-1][0:npeaks+1])
    
        peaks = []
        peak_region_top_right = []
        peak_region_xsize = []
        peak_region_ysize = []
        for kk in sorted_idx[::-1][0:npeaks+1]:
            print('areas[kk]:', kk, areas[kk])
            print('peak areas[kk]:', kk, peak_areas[kk])
            
            if kk == 0: continue
            if areas[kk] < 5: continue # too small: noise
            if peak_areas[kk] < 1: continue # too small: background
                
            idxs = np.where(label_image==kk)
            
            apex_intensity = xic_2d_max*image[idxs].max()
            
            print('max intensity in peak areas[kk]:', kk, apex_intensity)
            
            peak_region_top_right.append((idxs[1].min(),idxs[0].min()))
            peak_region_xsize.append(idxs[1].max()-idxs[1].min()+1)
            peak_region_ysize.append(idxs[0].max()-idxs[0].min()+1)
            
            subimg = image[idxs[0].min():idxs[0].max()+1,idxs[1].min():idxs[1].max()+1]
            
            lc_chrom = subimg.sum(axis=1).reshape((-1, 1))
            dt_chrom = subimg.sum(axis=0).reshape((-1, 1))
            
            lcidx = np.argmax(lc_chrom)
            dtidx = np.argmax(dt_chrom)
            
            print(idxs[0].min(),lcidx, idxs[1].min(),dtidx)
            peak_lc_idx = idxs[0].min()+lcidx
            peak_dt_idx = idxs[1].min()+dtidx
            peaks.append((peak_lc_idx, peak_dt_idx))
            
            local_peaks.append({'frame_idx':peak_lc_idx+_min_frame,
                                'scan_idx':peak_dt_idx+_min_scan,
                                'start_frame':idxs[0].min()+_min_frame,
                                'end_frame':idxs[0].max()+_min_frame,
                                'start_scan':idxs[1].min()+_min_scan,
                                'end_scan':idxs[1].max()+_min_scan,
                                'peak_area':peak_areas[kk],
                                'background_mad':background_mad,
                                'background_std':background_std,
                                'apex_intensity':apex_intensity})
            
            if len(peaks) >= npeaks: break
        peaks = np.array(peaks)
        
        if peaks.shape[0] == 0: continue #
        
        print('label_image:', label_image.shape)
        image_label_overlay = label2rgb(label_image, image=image)
        print('image_label_overlay:', image_label_overlay.shape)
        
        lc_chrom = image.sum(axis=1).reshape((-1, 1))
        dt_chrom = image.sum(axis=0).reshape((-1, 1))

        ##############################################
        plt.close('all')
        plt.figure(figsize=(10, 6))
        ax_dt = plt.subplot2grid((3, 8), (0, 0), colspan=7)
        ax_xic = plt.subplot2grid((3, 8), (1, 0), colspan=7, rowspan=2)
        ax_lc = plt.subplot2grid((3, 8), (1, 7), rowspan=2)
        
        ax_xic.imshow(image_label_overlay)
        ax_xic.scatter(peaks[:,1], peaks[:,0], c='r')
        
        for (p0,p1),w,h in zip(peak_region_top_right, peak_region_xsize, peak_region_ysize):
            print(p0+_min_scan,p1+_min_frame,w,h)
            # Add the patch to the Axes
            ax_xic.add_patch(patches.Rectangle((p0,p1),w,h,linewidth=4,edgecolor='r',facecolor='none'))

        ax_xic.set_yticks(np.arange(0, _max_frame-_min_frame, (_max_frame-_min_frame)//3))
        ax_xic.set_xticks(np.arange(0, _max_scan-_min_scan, (_max_scan-_min_scan)//10))
        ax_xic.set_xticklabels(np.array(ax_xic.get_xticks())+_min_scan)
        ax_xic.set_yticklabels(np.array(ax_xic.get_yticks())+_min_frame)
        
        base = ax_lc.transData
        rot = Affine2D().rotate_deg(270)

        ax_lc.plot(range(_min_frame,_max_frame), lc_chrom, transform= rot + base)
        ax_lc.scatter(peaks[:,0]+_min_frame, lc_chrom[peaks[:,0]], c='r', transform= rot + base)

        ax_dt.plot(range(_min_scan,_max_scan), dt_chrom)
        ax_dt.scatter(peaks[:,1]+_min_scan, dt_chrom[peaks[:,1]], c='r')
        
        plt.tight_layout()
        if fout is not None: plt.savefig(fout+"_frame={0}_scan={1}.png".format(peaks[0,0]+_min_frame,peaks[0,1]+_min_scan))
        else: plt.show()
        ##############################################
        
#         # image_max is the dilation of im with a 20*20 structuring element
#         # It is used within peak_local_max function
#         image_max = ndi.maximum_filter(image, size=5, mode='constant')

#         # Comparison between image_max and im to find the coordinates of local maxima
#         coordinates = peak_local_max(image, min_distance=3, num_peaks=2)

#         # display results
#         fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
#         ax = axes.ravel()
#         ax[0].imshow(image)
#         # ax[0].matshow(im, norm=LogNorm(vmin=10, vmax=500))
#         ax[0].axis('off')
#         ax[0].set_title('Original')

#         ax[1].imshow(image_max, cmap=plt.cm.gray)
#         ax[1].axis('off')
#         ax[1].set_title('Maximum filter')

#         ax[2].imshow(image, cmap=plt.cm.gray)
#         ax[2].autoscale(False)
#         ax[2].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
#         for p0,w,h in zip(peak_region_top_right, peak_region_xsize, peak_region_ysize):
#             print(p0,w,h)
#             # Add the patch to the Axes
#             ax[2].add_patch(patches.Rectangle(p0,w,h,linewidth=4,edgecolor='r',facecolor='none'))

#         ax[2].axis('off')
#         ax[2].set_title('Peak local max')

#         plt.show()
    return local_peaks


# find peaks in 2d image
def detect_peaks(image):
    """
    Takes an image and detect the peaks using the local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2,2)

    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    return detected_peaks


def read_pickle(file):
    with open(file, 'rb') as handle:
        obj = pickle.load(handle)
    return obj

def write_pickle(obj, file):
    with open(file, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    reader = UIMFReader(FLAGS.uimf, TIC_threshold=100)
    print("#scans:{0}, #frames:{1}".format(reader.num_scans, reader.num_frames))
    
    # collect mz_calibrators for all frames
    for f in range(1,reader.num_frames+1):
        mz_calibrator = reader.get_mz_calibrator(frame_num=f)
    print(reader.mz_calibrator_by_params)
    
    target_df = read_target_list(FLAGS.target, FLAGS.target_sheet_number)
    #print('Targets\n', target_df)
    target_mzbins, mzbins_by_mz = collect_target_mzbins(target_df, reader.mz_calibrator_by_params, ppm=FLAGS.ppm, isotope=FLAGS.isotope)
    
    if FLAGS.mode == 'binning':
        xic_by_mzbin = get_target_peaks(reader, target_mzbins, sig_th=FLAGS.sig_th, ofile=FLAGS.bin_file)
    elif FLAGS.mode == 'peak':
        xic_by_mzbin = read_pickle(FLAGS.bin_file)
        all_peaks = []
        for i,(seq,mono,mz,iso,z) in enumerate(mzbins_by_mz):
            print(i,(seq,mono,mz,iso,z))
            xic_2d, coordinates = get_roi(xic_by_mzbin, mzbins_by_mz[(seq,mono,mz,iso,z)], reader.num_frames, reader.num_scans,
                min_distance=30, num_peaks=5, normalize=False)
            local_peaks = peak_candidates(xic_2d, coordinates, frame_pad=100, scan_pad=400, npeaks=2, fout='xic_{0:.2f}_z{1:d}_iso{2:d}'.format(mz,z,iso))

            if len(local_peaks) > 0:
                local_peaks = pd.DataFrame(local_peaks).drop_duplicates()
                local_peaks['peptide_seq'] = seq
                local_peaks['mass'] = mono
                local_peaks['isotope_mass'] = mz
                local_peaks['isotope'] = iso
                local_peaks['charge'] = z
                all_peaks.append(local_peaks)
            #if i == 20: break
        pd.concat(all_peaks, ignore_index=True).to_csv('test.csv')

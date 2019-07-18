from uimfpy.UIMFReader import *
import numpy as np

from scipy.sparse import csc_matrix,csr_matrix,coo_matrix
import pickle

from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import data, img_as_float
from skimage.morphology import watershed
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb

from matplotlib.colors import LogNorm


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

def peak_candidates(xic_2d, coordinates, frame_pad=50, scan_pad=100, npeaks_per_candidate=3, masking_threshold=0.1, fout=None):
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
        image = im_gray[_min_frame:_max_frame,_min_scan:_max_scan]
        
        # filter by moving average
        moving_avg_image = ndi.uniform_filter(image, size=10, mode='constant')
        # filter by maximum (only for visualization)
        image_max = ndi.maximum_filter(image, size=10, mode='constant')
        # find peak candidates in moving_avg_image
        candidates = peak_local_max(moving_avg_image, min_distance=10, num_peaks=npeaks_per_candidate)
        
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
        for i in range(candidates.shape[0]):
            # label of a candidate
            target_label = label_im[candidates[i,0], candidates[i,1]]
            #print(candidates[i], target_label, sizes[target_label], peak_area[target_label])
            # if a region is meaningful
            if (sizes[target_label] > 0) & (target_label not in target_labels):
                target_labels.append(target_label)
                slice_x, slice_y = ndi.find_objects(label_im==target_label)[0]
                #print('boundary:', slice_x, slice_y)
                
                # save peaks
                local_peaks.append({'frame':candidates[i,0]+_min_frame, 'scan':candidates[i,1]+_min_scan,
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
            
            
        # display results
        #fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
        #ax = axes.ravel()
        #ax[0].imshow(image)
        #ax[0].axis('off')
        #ax[0].set_title('Original')

        #ax[1].imshow(image_max, cmap=plt.cm.gray)
        #ax[1].axis('off')
        #ax[1].set_title('Maximum filter')

        #ax[2].imshow(image, cmap=plt.cm.gray)
        #ax[2].autoscale(False)
        #ax[2].plot(candidates[:, 1], candidates[:, 0], 'r.')
        #ax[2].axis('off')
        #ax[2].set_title('Peak local max')

        #plt.tight_layout()
        ## plt.savefig('peak_local_max_{0:.2f}.png'.format(mz))
        #plt.show()
    #print(local_peaks)
    return local_peaks


from scipy.spatial.distance import cosine

def find_mzbin_infos(mzbins_by_mz, sequence, mono_mz, isotope, charge):
    for info in mzbins_by_mz:
        seq,_mz,mz,iso,z = info
        if (seq==sequence) & (z==charge) & (iso==isotope):
            return info
    return (sequence,mono_mz,mono_mz+isotope/charge, isotope, charge)

def get_2d(xic_2d, frame_start, frame_stop, scan_start, scan_stop):
    xic_2d_max = xic_2d.max()
    im_gray = xic_2d/xic_2d_max
    max_frames,max_scans = xic_2d.shape
#     image = im_gray[frame_start:frame_stop,scan_start:scan_stop]
#     image_2 = xic_2d[frame_start:frame_stop,scan_start:scan_stop]
#     image_2 = image_2/image_2.max()
#     plt.figure(figsize=(4, 2))
#     plt.axes([0, 0, 1, 1])
#     plt.imshow(image_2)
#     plt.axis('off')
#     plt.show()
    return xic_2d[frame_start:frame_stop,scan_start:scan_stop]

def collect_peaks(target_df, xic_by_mzbin, mzbins_by_mz, peak_out_file):
    heavy_only = True

    mzbins_list = [('KMPLDLS[+80]PLATPIIR[+10]', 877.979325, 877.979325, 0, 2),]
                   #('KMPLDLS[+80]PLATPIIR[+10]', 877.979325, 878.479325, 1, 2),
                   #('KMPLDLS[+80]PLATPIIR[+10]', 877.979325, 878.979325, 2, 2),
                   #('KMPLDLS[+80]PLATPIIR[+10]', 877.979325, 879.479325, 3, 2),
                   #('KMPLDLS[+80]PLATPIIR', 872.975191, 872.975191, 0, 2),
                   #('KMPLDLS[+80]PLATPIIR', 872.975191, 873.475191, 1, 2),
                   #('KMPLDLS[+80]PLATPIIR', 872.975191, 873.975191, 2, 2),
                   #('KMPLDLS[+80]PLATPIIR', 872.975191, 874.475191, 3, 2)]


    all_peaks = []
    # cache for the peak finding. No run for the same mzbins
    cache_local_peaks_by_mzbins = dict()

    stime = time.time()
    _num_mzbins = 0
    for info in mzbins_by_mz:
    # for info in mzbins_list:
        seq,_mz,mz,iso,z = info
        if heavy_only:
            if not seq.endswith(']'):
                #print(seq)
                continue
            if iso>0: continue
        #print(info)
        _num_mzbins+=1
        
        mzbins = mzbins_by_mz[info]
        
        mzbins_key = ''.join([str(m) for m in mzbins])
        # if there is a cache
        if mzbins_key in cache_local_peaks_by_mzbins:
            #print('found a cache:', mzbins_key)
            local_peaks = cache_local_peaks_by_mzbins[mzbins_key]
            if len(local_peaks) > 0:
                local_peaks_df = pd.DataFrame(local_peaks).drop_duplicates()
                local_peaks_df['pepseq'] = seq
                local_peaks_df['mono_mass'] = _mz
                local_peaks_df['iso_mass'] = mz
                local_peaks_df['iso'] = iso
                local_peaks_df['z'] = z
                all_peaks.append(local_peaks_df)
        else:
            if len(mzbins) == 0:
                #print("No mzbin:", info)
                continue

            im = xic_matrix(xic_by_mzbin, mzbins, reader.num_frames, reader.num_scans, normalize=False)
            #print(im.max())
            im_gray = im/im.max()

            # image_max is the dilation of im with a 20*20 structuring element
            # It is used within peak_local_max function
            #image_max = ndi.maximum_filter(im_gray, size=20, mode='constant')
            moving_avg_image = ndi.uniform_filter(im_gray, size=10, mode='constant')

            # Comparison between image_max and im to find the coordinates of local maxima
            coordinates = peak_local_max(moving_avg_image, min_distance=20, num_peaks=10)

            ##########################################################
            # logging 
            ##########################################################
    #         print(mzbins)
    #         for i in range(coordinates.shape[0]):
    #             _int = moving_avg_image[coordinates[i,0],coordinates[i,1]]
    #             print(coordinates[i], _int)
            ##########################################################
            # logging 
            ##########################################################

            local_peaks = peak_candidates(im, coordinates, frame_pad=50, scan_pad=100, npeaks_per_candidate=3)
            cache_local_peaks_by_mzbins[mzbins_key] = local_peaks
            if len(local_peaks) > 0:
                local_peaks_df = pd.DataFrame(local_peaks).drop_duplicates()
                local_peaks_df['pepseq'] = seq
                local_peaks_df['mono_mass'] = _mz
                local_peaks_df['iso_mass'] = mz
                local_peaks_df['iso'] = iso
                local_peaks_df['z'] = z
                all_peaks.append(local_peaks_df)

    #         # display results
    #         fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
    #         ax = axes.ravel()
    #         ax[0].imshow(im_gray)
    #         # ax[0].matshow(im, norm=LogNorm(vmin=10, vmax=500))
    #         ax[0].axis('off')
    #         ax[0].set_title('Original')

    #         ax[1].imshow(moving_avg_image, cmap=plt.cm.gray)
    #         ax[1].axis('off')
    #         ax[1].set_title('Maximum filter')

    #         ax[2].imshow(im_gray, cmap=plt.cm.gray)
    #         ax[2].autoscale(False)
    #         ax[2].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
    #         ax[2].axis('off')
    #         ax[2].set_title('Peak local max')

    #         plt.tight_layout()
    #         # plt.savefig('peak_local_max_{0:.2f}.png'.format(mz))

    #         plt.show()

    all_peaks_df = pd.concat(all_peaks, ignore_index=True)
    duration = time.time()-stime
    print(duration, 'sec', _num_mzbins, duration/_num_mzbins)
    #######################################################################################

    print('num_targets:', target_df.shape[0], target_df.shape[0]//2)
    num_peptides = target_df.shape[0]//2

    isotopes = 3

    all_peaks_cosine = []

    for i in range(num_peptides):
        light_df = target_df.iloc[2*i,:]
        heavy_df = target_df.iloc[2*i+1,:]
        
        light_pepseq = light_df['Modified Sequence']
        heavy_pepseq = heavy_df['Modified Sequence']
        light_mono_mz = light_df['Precursor Mz']
        heavy_mono_mz = heavy_df['Precursor Mz']
        charge = heavy_df['Precursor Charge']

        sequences = [light_pepseq,heavy_pepseq]
        temp_peaks = all_peaks_df[all_peaks_df.pepseq.isin(sequences)&(all_peaks_df.z==charge)]
        num_peaks = temp_peaks.shape[0]
        # print('num peaks:', num_peaks)
        #print(temp_peaks.head())

        if num_peaks>0:
            ############################################################
            # find available mzbins
            ############################################################
            heavy_mono_info = None
            for info in mzbins_by_mz:
                seq,_mz,mz,iso,z = info
                if (seq in sequences) & (charge==z):
                    if (seq==heavy_pepseq) & (iso==0):
                        heavy_mono_info = info
                    #else:
                    #    mzbins_info_list.append(info)
            #print('heavy mono:', heavy_mono_info)
            #print(mzbins_info_list)
            ############################################################
            # 
            ############################################################
            if heavy_mono_info is None: continue
                
            #heavy_mono_info = (heavy_pepseq, heavy_mono_mz, heavy_mono_mz, 0, charge)
            heavy_mono_mzbins = mzbins_by_mz[heavy_mono_info]
            if len(heavy_mono_mzbins) == 0:
                print("No mzbins for heavy peptide")
            else:
                ############################################################
                # find available mzbins
                ############################################################
                mzbins_list = []
                for iso in range(isotopes+1):
                    if iso > 0: mzbins_list.append(find_mzbin_infos(mzbins_by_mz, heavy_pepseq, heavy_mono_mz, iso, charge))
                    mzbins_list.append(find_mzbin_infos(mzbins_by_mz, light_pepseq, light_mono_mz, iso, charge))
                #print(mzbins_list)
                ############################################################
                # find available mzbins
                ############################################################

                xic2d = xic_matrix(xic_by_mzbin, heavy_mono_mzbins, reader.num_frames, reader.num_scans, normalize=False)
                cos_arr = []
                for row in temp_peaks.to_dict('records'):
                    #print(row)
                    ############################################################
                    # compute cosine similarity for each peak region
                    ############################################################
                    # heavy mono vector
                    heavy_vec = get_2d(xic2d, row['frame_start'], row['frame_end'], row['scan_start'], row['scan_end'])
                    heavy_lc_chrom = heavy_vec.sum(axis=1)
                    heavy_mobility = heavy_vec.sum(axis=0)
                    heavy_area = heavy_vec.sum()
                    light_area = 0
                    xic_cosine_sim = []
                    lc_cosine_sim = []
                    heavy_light_ratio = 0
                    for info in mzbins_list:
                        if info not in mzbins_by_mz: continue
                        _mzbins = mzbins_by_mz[info]
                        if len(_mzbins) > 0:
                            _xic2d = xic_matrix(xic_by_mzbin, _mzbins, reader.num_frames, reader.num_scans, normalize=False)
                            _vec = get_2d(_xic2d, row['frame_start'], row['frame_end'], row['scan_start'], row['scan_end'])
                            _lc_chrom = _vec.sum(axis=1)
                            _mobility = _vec.sum(axis=0)
                            # compute cosine distance
                            _xic_cosine_sim = 1-cosine(heavy_vec.flatten(), _vec.flatten())
                            _lc_cosine_sim = 1-cosine(heavy_lc_chrom, _lc_chrom)
                            #print(info, _mzbins, _cosine_sim)
                            if not np.isnan(_xic_cosine_sim):
                                xic_cosine_sim.append(_xic_cosine_sim)
                                lc_cosine_sim.append(_lc_cosine_sim)
                            if (info[0]==light_pepseq)&(info[3]==0):
                                light_area = _vec.sum()
                                heavy_light_ratio = heavy_area/light_area
                                heavy_light_lc_cosine = _lc_cosine_sim
                    #print(cosine_sim)
                    row['xic_cosine'] = np.sum(xic_cosine_sim)/(2*isotopes+1)
                    row['lc_cosine'] = np.sum(lc_cosine_sim)/(2*isotopes+1)
                    row['light_area'] = light_area
                    row['heavy_light_ratio'] = heavy_light_ratio
                    row['heavy_light_lc_cosine'] = heavy_light_lc_cosine
                    row['light_mono_mz'] = light_mono_mz
                    row['light_pepseq'] = light_pepseq
                    if row['xic_cosine'] > 0: all_peaks_cosine.append(row)

    all_peaks_cosine = pd.DataFrame(all_peaks_cosine)
    all_peaks_cosine[all_peaks_cosine.iso==0].to_csv(peak_out_file)

#######################################################################################
# uimf = 'data/LC_HEAVY_AML_F80000V1v1O1c5000Ct5NF5000V12v12N_1_MA_comp5.uimf'
# reader = UIMFReader(uimf, TIC_threshold=0)
# bins = reader.get_mz_peaks(frame_nums=2000)
# print(reader.mz_calibrator_by_params)
# target_df = read_target_list('data/20190319-Slim charge1-5 light-heavy.xlsx', sheet_name=0)
# print('target_df:', target_df.shape)
# target_mzbins, mzbins_by_mz = collect_target_mzbins(target_df, reader.mz_calibrator_by_params, ppm=40, isotope=3, heavy_only=False)

# mzbin_file = 'data/LC_HEAVY_AML_F80000V1v1O1c5000Ct5NF5000V12v12N_1_MA_comp5_50ppm.pkl'
# with open(mzbin_file, 'rb') as handle:
#     xic_by_mzbin = pickle.load(handle)
    
# peak_out_file = 'data/LC_HEAVY_AML_F80000V1v1O1c5000Ct5NF5000V12v12N_1_MA_comp5_50ppm_peak_list.csv'

# collect_peaks(target_df, xic_by_mzbin, mzbins_by_mz, peak_out_file)
#######################################################################################

for _i in [0, 78, 156, 312, 625, 1250, 2500, 5000]:
    print("#"*100)
    stime = time.time()
    uimf = 'data/LC_{0}_heavy_repA_1_MinInt5_DComp5.uimf'.format(_i)
    reader = UIMFReader(uimf, TIC_threshold=0)
    bins = reader.get_mz_peaks(frame_nums=2000)
    print('# ready uimf: {},'.format(uimf), (time.time()-stime)/60, 'min')
    target_df = read_target_list('data/20190319-Slim charge1-5 light-heavy.xlsx', sheet_name=0)
    print('# ready targets:{0},'.format(target_df.shape[0]), (time.time()-stime)/60, 'min')
    target_mzbins, mzbins_by_mz = collect_target_mzbins(target_df, reader.mz_calibrator_by_params, ppm=40, isotope=3, heavy_only=False)
    print('# ready target m/z bins,', (time.time()-stime)/60, 'min')
    mzbin_file = 'data/LC_{0}_heavy_repA_1_MinInt5_DComp5_50ppm.pkl'.format(_i)
    with open(mzbin_file, 'rb') as handle:
        xic_by_mzbin = pickle.load(handle)
    print('# ready xic_by_mzbin file,', (time.time()-stime)/60, 'min')
    peak_out_file = 'data/LC_{0}_heavy_repA_1_MinInt5_DComp5_50ppm_peak_list.csv'.format(_i)
    collect_peaks(target_df, xic_by_mzbin, mzbins_by_mz, peak_out_file)
    print('# Done,', (time.time()-stime)/60, 'min')

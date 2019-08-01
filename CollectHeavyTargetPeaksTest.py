from CollectHeavyTargetPeaks import collect_peaks, find_mzbin_infos

def test_collect_peaks(uimf_reader, xic_by_mzbins, mzbins_by_mz, pepseq, z, num_frames, num_scans,
                       global_min_distance, local_min_distance,
                       num_top_peaks, num_global_peaks, frames_for_peak, scans_for_peak,
                       masking_threshold, th_peak_area, moving_avg_size, debug=True):
    mono_info = find_mzbin_infos(mzbins_by_mz, pepseq, 0, z)
    if debug: print("find mzbins by sequence", mono_info)
    target_mzbin = mzbins_by_mz[mono_info]
    local_peaks = collect_peaks(uimf_reader, xic_by_mzbins, target_mzbin, mono_info, num_frames, num_scans,
                                global_min_distance=global_min_distance, local_min_distance=local_min_distance,
                                num_top_peaks=num_top_peaks, num_global_peaks=num_global_peaks,
                                frames_for_peak=frames_for_peak, scans_for_peak=scans_for_peak,
                                masking_threshold=masking_threshold, th_peak_area=th_peak_area,
                                moving_avg_size=moving_avg_size, debug=debug)
    return local_peaks
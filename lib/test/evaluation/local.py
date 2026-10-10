from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/dfc/Documents/dwh_documents/TSMTrack/data/got10k_lmdb'
    settings.got10k_path = '/home/dfc/Documents/dwh_documents/TSMTrack/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/dfc/Documents/dwh_documents/TSMTrack/data/itb'
    settings.lasot_extension_subset_path_path = '/home/dfc/Documents/dwh_documents/TSMTrack/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/dfc/Documents/dwh_documents/TSMTrack/data/lasot_lmdb'
    settings.lasot_path = '/home/dfc/Documents/dwh_documents/TSMTrack/data/lasot'
    settings.network_path = '/home/dfc/Documents/dwh_documents/TSMTrack/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/dfc/Documents/dwh_documents/TSMTrack/data/nfs'
    settings.otb_path = '/home/dfc/Documents/dwh_documents/TSMTrack/data/otb'
    settings.prj_dir = '/home/dfc/Documents/dwh_documents/TSMTrack'
    settings.result_plot_path = '/home/dfc/Documents/dwh_documents/TSMTrack/output/test/result_plots'
    settings.results_path = '/home/dfc/Documents/dwh_documents/TSMTrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/dfc/Documents/dwh_documents/TSMTrack/output'
    settings.segmentation_path = '/home/dfc/Documents/dwh_documents/TSMTrack/output/test/segmentation_results'
    settings.tc128_path = '/home/dfc/Documents/dwh_documents/TSMTrack/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/dfc/Documents/dwh_documents/TSMTrack/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/dfc/Documents/dwh_documents/TSMTrack/data/trackingnet'
    settings.uav_path = '/home/dfc/Documents/dwh_documents/TSMTrack/data/uav'
    settings.vot18_path = '/home/dfc/Documents/dwh_documents/TSMTrack/data/vot2018'
    settings.vot22_path = '/home/dfc/Documents/dwh_documents/TSMTrack/data/vot2022'
    settings.vot_path = '/home/dfc/Documents/dwh_documents/TSMTrack/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings


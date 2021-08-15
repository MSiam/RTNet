import os
import glob

val_seq = ['arabian_horn_viper', 'arctic_fox_1', 'arctic_wolf_1', 'black_cat_1', 'crab', 'crab_1',
            'cuttlefish_0', 'cuttlefish_1', 'cuttlefish_4', 'cuttlefish_5',
            'devil_scorpionfish', 'devil_scorpionfish_1', 'flatfish_2', 'flatfish_4', 'flounder',
            'flounder_3', 'flounder_4', 'flounder_5', 'flounder_6', 'flounder_7',
            'flounder_8', 'flounder_9', 'goat_1', 'hedgehog_1', 'hedgehog_2', 'hedgehog_3',
            'hermit_crab', 'jerboa', 'jerboa_1', 'lion_cub_0', 'lioness', 'marine_iguana',
            'markhor', 'meerkat', 'mountain_goat', 'peacock_flounder_0',
            'peacock_flounder_1', 'peacock_flounder_2', 'polar_bear_0', 'polar_bear_2',
            'scorpionfish_4', 'scorpionfish_5', 'seal_1', 'shrimp',
            'snow_leopard_0', 'snow_leopard_1', 'snow_leopard_2', 'snow_leopard_3', 'snow_leopard_6',
            'snow_leopard_7', 'snow_leopard_8', 'spider_tailed_horned_viper_0',
            'spider_tailed_horned_viper_2', 'spider_tailed_horned_viper_3',
            'arctic_fox', 'arctic_wolf_0', 'devil_scorpionfish_2', 'elephant',
            'goat_0', 'hedgehog_0',
            'lichen_katydid', 'lion_cub_3', 'octopus', 'octopus_1',
            'pygmy_seahorse_2', 'rodent_x', 'scorpionfish_0', 'scorpionfish_1',
            'scorpionfish_2', 'scorpionfish_3', 'seal_2',
            'bear', 'black_cat_0', 'dead_leaf_butterfly_1', 'desert_fox', 'egyptian_nightjar',
            'pygmy_seahorse_4', 'seal_3', 'snowy_owl_0',
            'flatfish_0', 'flatfish_1', 'fossa', 'groundhog', 'ibex', 'lion_cub_1', 'nile_monitor_1',
            'polar_bear_1', 'spider_tailed_horned_viper_1']

data_dir = '/local/riemann/home/msiam/MoCA_filtered2/'
img_dir = os.path.join(data_dir, 'JPEGImages/')
samples = []
for seq in val_seq:
    samples += sorted(glob.glob(os.path.join(img_dir, seq, '*.jpg')))

with open(os.path.join(data_dir, 'val.txt'), 'w') as f :
    for sample in samples:
        sample = sample.replace(data_dir, '/')
        f.write('%s %s\n'%(sample, sample))

import pandas as pd

def prepare_env():
    world = pd.read_csv("data/sPlotOpen/worldclim_data.csv")
    soil = pd.read_csv("data/sPlotOpen/soilgrid_data.csv")
    world = np.array(world[['bio_1', 'bio_2', 'bio_3', 'bio_4', 'bio_5', 'bio_6',
           'bio_7', 'bio_8', 'bio_9', 'bio_10', 'bio_11', 'bio_12', 'bio_13',
           'bio_14', 'bio_15', 'bio_16', 'bio_17', 'bio_18', 'bio_19']])
    soil = np.array(soil[['ORCDRC', 'PHIHOX', 'CECSOL', 'BDTICM', 'CLYPPT',
           'SLTPPT', 'SNDPPT', 'BLDFIE']])
    env = np.concatenate((world, soil), axis = 1)

    np.save("data/sPlotOpen/env.npy", env)


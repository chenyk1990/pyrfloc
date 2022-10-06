## The first example of the pyrfloc package
##
## Please download data into the ./data folder in your running directory
## Datapath: 
## https://drive.google.com/drive/folders/1-Pwa41_3C97XamG7NLVyMlOF9yArYcho?usp=sharing
## 
## Note: 
## times.bin can be re-generated using pyekfmm and the vpmod.bin model
## It was originally generated in the directory ~/research/eqsource_ml/db1d_syn9_v2/

import pyrfloc as loc

fpicks='./data/stations_picks.txt'
tfile='./data/times.bin'
vpfile='./data/vpmod.bin'
evfile='./data/locations_10000.txt'

## initialization
myloc=loc.RFloc3D(fpicks=fpicks);

## plot stations
myloc.plot_stations(evs=[-104.05,31.7,7.1]); #evs is the catalog locations

## load synthetic traveltimes
myloc.load_traveltime(tfile=tfile);

## load vp model
myloc.load_vel(vpfile=vpfile);

## plot vel
myloc.plot_vel();

## load synthetic event locations
myloc.load_events(evfile=evfile);

## using pyekfmm to calculate traveltime
# shots=myloc.evlabels.transpose();
# myloc.calc_traveltime(shots);
# myloc.stimes=myloc.fmmtimes;

## sample traveltime data on stations and prepare training data
myloc.prepare_data();
myloc.prepare_train_data();

## train
myloc.train();

## check train
myloc.train_check();

## save model
myloc.save_model(filename='./model.joblib');

## load model
myloc.load_model(filename='./model.joblib');

## real location
myloc.apply();

## plot final result
myloc.plot_result(evs=[-104.05,31.7,7.1]);








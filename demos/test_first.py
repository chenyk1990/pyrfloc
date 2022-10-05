import pyrfloc as loc

fpicks='/Users/chenyk/chenyk/eqsource_ml/syn9/matfun/stations_locations_times_39.txt'
tfile='/Users/chenyk/chenyk/eqsource_ml/db1d_syn9_v2/times_vnew1.1.bin'
vpfile='/Users/chenyk/chenyk/eqsource_ml/db1d_syn9_v2/vpmod.bin'
evfile='/Users/chenyk/chenyk/eqsource_ml/db1d_syn9_v2/locations_10000.txt'

## initialization
myloc=loc.RFloc3D(fpicks=fpicks);

## plot stations
myloc.plot_stations(evs=[-104.05,31.7,7.1]);

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








export OMP_NUM_THREADS=18

../../jagurs_test par=tsun.par

mkdir snap_grd
mkdir tgsfiles
mv SD01.*.grd snap_grd/.
mv tgs0* tgsfiles/.

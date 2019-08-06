#!/bin/bash
filter_type=$1
L=$2
obs_err=$3
N=$4
F=$5
ROI=$6
alpha=$7
dk=$8

outdir="/glade/scratch/mying/L96_DA/${filter_type}_dk$dk/L${L}_s${obs_err}/N${N}_F${F}/ROI${ROI}_relax$alpha"

cat > tmp.sh << EOF
#!/bin/bash
#PBS -A P54048000
#PBS -N L96
#PBS -l walltime=5:00:00
#PBS -l select=1:ncpus=1
#PBS -q regular
#PBS -j oe
#PBS -o log
mkdir -p $outdir
cd /glade/work/mying/L96_DA
./run_cycle_filter.py $filter_type $L $obs_err $N $F $ROI $alpha $dk
./diag.py $outdir $L $obs_err $dk
EOF

#qsub tmp.sh
#rm tmp.sh

#source /net/lem/data1/regt/activate_env.sh
cd /net/lem/data2/regt/pRT_opacities/

nice -n 10 python -u main.py -cs -i_min 40 -i_max 54 >& logs/CH4_40_54.out
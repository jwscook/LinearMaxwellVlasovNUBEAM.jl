#  Shot   fci    Bt  ne (e13 cm-3) Zeff nh (e13 cm-3) va (m/s)
#  184057 15.85 2.07 3.3e13        1.14 3.05          7.6e6
#  184061 9.89  1.29 2.9e13        1.38 2.37          4.8e6

# Shot  Te (keV)  Ti (keV)  nb/ne
# 184057 1.31 0.90 0.07
# 184061 1.68 0.92 0.12

#julia -t 6 --proj src/ICE2D.jl --tlh 0.1 --niters 1 --nrbs 1\
#    --nubeamfilename fnb_184061H02_fi_1_scaled.h5\
#    --te 1680 --tp 920 --nb_ne 0.12 --ne 2.9e19 --np 2.37e19 --B0=1.29 --Zeff=1.38\
#    --otheralfvenspeed 4.8e6

#for vthfracinj in 0.0 0.0001 0.001 0.01 0.02 0.04 0.06 0.08 0.1; do
#  julia -t 6 --proj src/ICE2D.jl --tlh 0.1 --niters 1 --nrbs 1\
#      --nubeamfilename fnb_184061H02_fi_1_scaled.h5\
#      --te 1680 --tp 920 --nb_ne 0.12 --ne 2.9e19 --np 2.37e19 --B0=1.29 --Zeff=1.38\
#      --otheralfvenspeed 4.8e6 --vthfracinj $vthfracinj
#  mkdir -p primary_vthfracinj$vthfracinj
#  rm primary_vthfracinj$vthfracinj/*.pdf
#  rm primary_vthfracinj$vthfracinj/*.jld
#  mv *vthfracinj,$vthfracinj* primary_vthfracinj$vthfracinj
#  mv *vthfracinj,$vthfracinj* primary_vthfracinj$vthfracinj
#done


#
##is=( 0.25 1.0 )
##js=( 32 128 )
is=( 0.5) # 1.0 0.0 0.0 0.0
js=( 64 ) # 64  1   16  32
ks=( 4  ) # 4   0   0   0
for idx in "${!is[@]}"; do
  tlh=${is[$idx]}
  nrbs=${js[$idx]}
  niters=${ks[$idx]}
  nb_ne=0.12
  echo "--tlh =" $tlh
  echo "--nrbs =" $nrbs
  echo "--nb_ne =" $nb_ne
  julia -t 8 --proj src/ICE2D.jl --tlh $tlh --niters $niters --nrbs $nrbs\
    --nubeamfilename fnb_184061H02_fi_1_scaled.h5\
    --te 1680 --tp 920 --nb_ne 0.12 --ne 2.9e19 --np 2.37e19 --B0=1.29 --Zeff=1.38\
    --otheralfvenspeed 4.8e6
  #for nb_ne in 0.012 0.0012 0.00012
  #do
  #  echo "--nb_ne =" $nb_ne
  #  julia -t 6 --proj src/ICE2D.jl --tlh 0.0 --niters 0 --nrbs $nrbs\
  #    --nubeamfilename fnb_184061H02_fi_1_scaled.h5\
  #    --te 1680 --tp 920 --nb_ne $nb_ne --ne 2.9e19 --np 2.37e19 --B0=1.29 --Zeff=1.38\
  #    --otheralfvenspeed 4.8e6
  #done
done
#

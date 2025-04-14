# tohoku.grd
wget https://kiyuu.bosai.go.jp/GtTM/TwDB/eastJapan.grd
gmt grdsample eastJapan.grd -R140.5/146.5/34.0/42.0 -I0.005/0.005 -Gtohoku.grd=cf
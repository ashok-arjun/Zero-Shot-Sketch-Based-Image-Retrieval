echo "Downloading the Sketchy dataset (it will take some time)"
python3 src/download_gdrive.py 0B7ISyeE8QtDdTjE1MG9Gcy1kSkE $path_dataset/Sketchy.7z
echo -n "Unzipping it..."
7z x $path_dataset/Sketchy.7z -o$path_dataset > $path_dataset/garbage.txt
echo "Done"
rm $path_dataset/garbage.txt
rm $path_dataset/Sketchy.7z
rm $path_dataset/README.txt
mv $path_dataset/256x256 $path_dataset/Sketchy
echo "Downloading the extended photos of Sketchy dataset (it will take some time)"
python3 src/download_gdrive.py 0B2U-hnwRkpRrdGZKTzkwbkEwVkk $path_dataset/Sketchy/extended_photo.zip
echo -n "Unzipping it..."
unzip -qq $path_dataset/Sketchy/extended_photo.zip -d $path_dataset/Sketchy
rm $path_dataset/Sketchy/extended_photo.zip
mv $path_dataset/Sketchy/EXTEND_image_sketchy $path_dataset/Sketchy/extended_photo
rm -r $path_dataset/Sketchy/sketch/tx_000000000010
rm -r $path_dataset/Sketchy/sketch/tx_000000000110
rm -r $path_dataset/Sketchy/sketch/tx_000000001010
rm -r $path_dataset/Sketchy/sketch/tx_000000001110
rm -r $path_dataset/Sketchy/sketch/tx_000100000000
rm -r $path_dataset/Sketchy/photo/tx_000100000000
mv $path_dataset/Sketchy/sketch/tx_000000000000/hot-air_balloon $path_dataset/Sketchy/sketch/tx_000000000000/hot_air_balloon
mv $path_dataset/Sketchy/sketch/tx_000000000000/jack-o-lantern $path_dataset/Sketchy/sketch/tx_000000000000/jack_o_lantern
mv $path_dataset/Sketchy/photo/tx_000000000000/hot-air_balloon $path_dataset/Sketchy/photo/tx_000000000000/hot_air_balloon
mv $path_dataset/Sketchy/photo/tx_000000000000/jack-o-lantern $path_dataset/Sketchy/photo/tx_000000000000/jack_o_lantern
mv $path_dataset/Sketchy/extended_photo/hot-air_balloon $path_dataset/Sketchy/extended_photo/hot_air_balloon
mv $path_dataset/Sketchy/extended_photo/jack-o-lantern $path_dataset/Sketchy/extended_photo/jack_o_lantern
echo "Done"
echo "Sketchy dataset is now ready to be used"

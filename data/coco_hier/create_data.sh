cur_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )
root_dir=$cur_dir/../..

cd $root_dir

redo=1
data_root_dir="/home/data/liuyong/choas/datasets/MSCOCO/coco"
dataset_name="select299"
mapfile="$root_dir/data/$dataset_name/labelmap_select.prototxt"
anno_type="detection"
label_type="json"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0

extra_cmd="--encode-type=jpg --encoded"
if [ $redo ]
then
  extra_cmd="$extra_cmd --redo"
fi
for subset in train_coco_imagenet_det
do
  python $root_dir/scripts/create_annoset.py --anno-type=$anno_type --label-type=$label_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --shuffle --check-label $extra_cmd $data_root_dir $root_dir/data/$dataset_name/$subset.txt data/$dataset_name/$db/$dataset_name"_"$subset"_"$db
done

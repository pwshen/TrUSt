# cmd_template = "srun --account=MST107266 --gres=gpu:1 --cpus-per-task=4 python main.py --feat_dir /home/ganymede9487/video_trope/data --bert_dir /work/ganymede9487/video_trope/bert/ --cnn_feat /work/ganymede9487/video_trope/trope_hero_resnet.h5 --c3d_feat /work/ganymede9487/video_trope/trope_hero_slowfast.h5 {} {} {}"
cmd_template = "srun --account=MST107266 --gres=gpu:1 --cpus-per-task=4 python3 -u multitaskmain.py --feat_dir /work/pwshen1214/data --bert_dir /work/pwshen1214/data/bert/ --vcpt_dir /work/pwshen1214/tvtrope_vcpts/ --cnn_feat /work/pwshen1214/data/trope_hero_resnet.h5 --c3d_feat /work/pwshen1214/data/trope_hero_slowfast.h5 --sn_feat /work/pwshen1214/data/TVtrope_soundnet.hdf5 {} {} {} {} {}"

opts = [
    ['vid', 'cnn', '', '', ''],
    ['vid', 'c3d', '', '', ''],
    ['vid', 'both', '', '', ''],
    ['sub', '', '', '', ''],
    ['aud', '', '', 'sn', ''],
    ['obj', '', '', '', 'frcnn'],
    #one feature only
    ['vid sub', 'cnn', 'sub', '', ''],
    ['vid sub', 'c3d', 'sub', '', ''],
    ['vid sub', 'both', 'sub', '', ''],
    ['vid aud', 'cnn', '', 'sn', ''],
    ['vid aud', 'c3d', '', 'sn', ''],
    ['vid aud', 'both', '', 'sn', ''],
    #visual audio
    ['vid obj', 'cnn', '', '', 'frcnn'],
    ['vid obj', 'c3d', '', '', 'frcnn'],
    ['vid obj', 'both', '', '', 'frcnn'],
    #visual object
    ['sub obj', '', 'sub', '', 'frcnn'],
    ['aud obj', '', '', 'sn', 'frcnn'],
    #audio object
    ['vid sub obj', 'cnn', 'sub', '', 'frcnn'],
    ['vid sub obj', 'c3d', 'sub', '', 'frcnn'],
    ['vid sub obj', 'both', 'sub', '', 'frcnn'],
    ['vid aud obj', 'cnn', '', 'sn', 'frcnn'],
    ['vid aud obj', 'c3d', '', 'sn', 'frcnn'],
    ['vid aud obj', 'both', '', 'sn', 'frcnn'],
    #video audio object
    ['vid cap', 'cnn', '', '', ''],
    ['vid cap', 'c3d', '', '', ''],
    ['vid cap', 'both', '', '', ''],
    ['vid sub cap', 'cnn', 'sub', '', ''],
    ['vid sub cap', 'c3d', 'sub', '', ''],
    ['vid sub cap', 'both', 'sub', '', ''],
    ['vid aud cap', 'cnn', '', 'sn', ''],
    ['vid aud cap', 'c3d', '', 'sn', ''],
    ['vid aud cap', 'both', '', 'sn', ''],
    ['vid obj cap', 'cnn', '', '', 'frcnn'],
    ['vid obj cap', 'c3d', '', '', 'frcnn'],
    ['vid obj cap', 'both', '', '', 'frcnn'],
    ['vid sub obj cap', 'cnn', 'sub', '', 'frcnn'],
    ['vid sub obj cap', 'c3d', 'sub', '', 'frcnn'],
    ['vid sub obj cap', 'both', 'sub', '', 'frcnn'],
    ['vid aud obj cap', 'cnn', '', 'sn', 'frcnn'],
    ['vid aud obj cap', 'c3d', '', 'sn', 'frcnn'],
    ['vid aud obj cap', 'both', '', 'sn', 'frcnn'],
    #all
    ['vid cap', 'cnn', '', '', '', 'attn'],
    ['vid cap', 'c3d', '', '', '', 'attn'],
    ['vid cap', 'both', '', '', '', 'attn'],
    ['vid sub cap', 'cnn', 'sub', '', '', 'attn'],
    ['vid sub cap', 'c3d', 'sub', '', '', 'attn'],
    ['vid sub cap', 'both', 'sub', '', '', 'attn'],
    ['vid aud cap', 'cnn', '', 'sn', '', 'attn'],
    ['vid aud cap', 'c3d', '', 'sn', '', 'attn'],
    ['vid aud cap', 'both', '', 'sn', '', 'attn'],
    ['vid obj cap', 'cnn', '', '', 'frcnn', 'attn'],
    ['vid obj cap', 'c3d', '', '', 'frcnn', 'attn'],
    ['vid obj cap', 'both', '', '', 'frcnn', 'attn'],
    ['vid sub obj cap', 'cnn', 'sub', '', 'frcnn', 'attn'],
    ['vid sub obj cap', 'c3d', 'sub', '', 'frcnn', 'attn'],
    ['vid sub obj cap', 'both', 'sub', '', 'frcnn', 'attn'],
    ['vid aud obj cap', 'cnn', '', 'sn', 'frcnn', 'attn'],
    ['vid aud obj cap', 'c3d', '', 'sn', 'frcnn', 'attn'],
    ['vid aud obj cap', 'both', '', 'sn', 'frcnn', 'attn']
    #all+attn
]

for i, opt in enumerate(opts):
    opt1 = "--input_streams {}".format(opt[0])
    opt2 = "--visual_feat {}".format(opt[1]) if opt[1] else ""
    opt3 = "--speech_feat {}".format(opt[2]) if opt[2] else ""
    opt4 = "--audio_feat {}".format(opt[3]) if opt[3] else ""
    opt5 = "--object_feat {}".format(opt[4]) if opt[4] else ""
    cmd = cmd_template.format(opt1, opt2, opt3, opt4, opt5)
    if len(opt)==5:
        cmd += ' --disable_attn_fusion'
    cmd_5 = " & ".join(["{}  --trope_file trope_split{}.json".format(cmd, j) for j in range(1,6)])
    fn = "exp{}_({}).sh".format(i, ")(".join(opt))
    with open(fn, "w") as f:
        f.write(cmd_5)
    #print(fn)
    #print(cmd_5)
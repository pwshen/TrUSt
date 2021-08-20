cmd_template = "python3 main.py --feat_dir ./data --bert_dir ./data/bert/ --vcpt_dir ./data/tvtrope_vcpts/ --cnn_feat ./data/TVtrope_resnet101.hdf5 --s3d_feat ./data/truman_s3d.h5 --sn_feat ./data/TVtrope_soundnet.hdf5 {} {} {} {} {}"

opts = [
    ['vid sub cap', 's3d', 'sub', '', ''],
    ['vid aud cap', 's3d', '', 'sn', ''],
    ['vid obj cap', 's3d', '', '', 'frcnn'],
    ['vid sub obj cap', 's3d', 'sub', '', 'frcnn'],
    ['vid aud obj cap', 's3d', '', 'sn', 'frcnn'],
    #all
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

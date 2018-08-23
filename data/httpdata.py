import tensorflow as tf

dataset = tf.data.TextLineDataset('ftp://storage.googleapis.com/kaggle-competitions-data/kaggle/3136/train.csv?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1534590218&Signature=Pb52T9Rzk%2F15nTNAuHhx%2BQcIPCBBUCSKHfaSNHRTEZo2uDqam25xLVonddhnK6KQ%2FwGhxbvT05NWAceW3c8tJxxbZv8MfURRcJhor%2FpeDVb1ZQd0hUUlKkB%2Bb9bDTA%2Bi59JSr6XT7qaS5In3BY4MZds7l6iEQI3duzsMB1DUX443swVRZkRFehfE6yGoTh5zVsnBbHoEOu0%2BPdTE5pRB5b%2FyYU2gUefCCYR9tn%2BbtsIw211szGWOST%2BvIz%2BamK1GG3ohrshw%2FONoYvX3UhzU46i05KvmeieinCXYb%2Fn0ndd%2B3rNojXCLsGTeaTRlxED%2F5GgSE0AFGFlSWDnnerVS%2Fg%3D%3D')
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    for i in range(100):
        value = sess.run(next_element)
        print(value)


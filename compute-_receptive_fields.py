
#!/usr/bin/env python

import argparse

def recep_recur(output_size, ksize, stride):
    return (output_size - 1) * stride + ksize

def print_receptive_fields(params):
    for x in xrange(len(params)):
        name, cur_ksize, cur_stride = params[x]
        output_size = 1
        for reverse_iter in xrange(x, -1, -1):
            __, ksize, stride = params[reverse_iter]
            ksize, stride = int(ksize), int(stride)
            output_size = recep_recur(output_size, ksize, stride)
        print "{}:\trf={}\tksize={}\tstride={}".format(
            name, output_size, int(cur_ksize), int(cur_stride))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('param_file', type=str,
        help='''File containing kernel sizes and stride sizes
                of each layer.''')
    args = parser.parse_args()

    params = []
    with open(args.param_file, 'r') as f:
        for line in f:
            # k_sizes = [int(i) for i in line.split()]
            cur_param = line.split()
            params.append(cur_param)

    print_receptive_fields(params)
